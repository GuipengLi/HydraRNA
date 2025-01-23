from torch.utils.data import Dataset, Subset
import os
import sys
from pathlib import Path
from sklearn import metrics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from sklearn.metrics import precision_recall_curve, auc, mean_squared_error
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef

from collections import defaultdict


from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, pearsonr

from torch.nn.utils.weight_norm import weight_norm

from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data.data_utils import collate_tokens


from torch.amp import GradScaler




def _outer_concat(t1: torch.Tensor, t2: torch.Tensor):
    # t1, t2: shape = B x L x E
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"

    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

    return torch.concat((a, b), dim=-1)

class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x
        x = self.conv_net(x)
        x = x + residual
        return x

class ResNet2D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

## this prediction head is from https://github.com/lbcb-sci/RiNALMo/blob/main/rinalmo/model/downstream.py
class SecStructPredictionHead(nn.Module):
    def __init__(self, embed_dim, num_blocks, conv_dim=128, kernel_size=7):
        super().__init__()

        self.linear_in = nn.Linear(embed_dim * 2, conv_dim)
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")
        
    def forward(self, x):
        x = _outer_concat(x, x) # B x L x F => B x L x L x 2F

        x = self.linear_in(x)
        x = x.permute(0, 3, 1, 2) # B x L x L x E  => B x E x L x L

        x = self.resnet(x)
        x = self.conv_out(x)
        x = x.squeeze(-3) # B x 1 x L x L => B x L x L

        # Symmetrize the output
        x = torch.triu(x, diagonal=1)
        x = x + x.transpose(-1, -2)

        return x


parser = options.get_generation_parser(default_task='masked_lm_span')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path='../weights/HydraRNA_model.pt' # change the path if necessary
args = options.parse_args_and_arch(parser,  ['../dict/'])

# Setup task
task = tasks.setup_task(args)
print('| loading model from {}'.format(path))

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

input_dim = 1024
hidden_dim = 128
output_dim = 2

num_epochs = 20

batch_size = 1
accumulation_steps = 4

MODEL = 'finetune_HydraAttRNA12_RNASecStruct12'


from rinalmo.utils.sec_struct import parse_sec_struct_file
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct, ss_precision, ss_recall, ss_f1, save_to_ct


class SecondaryStructureDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        min_seq_len: int = 0,
        max_seq_len: int = 10240,
        ss_file_extensions: list[str] = ["ct", "bpseq", "st"],
    ):
        super().__init__()

        self.data_dir = Path(data_dir)

        # Collect secondary structure file paths
        self.ss_paths = []
        for ss_file_ext in ss_file_extensions:
            for ss_file_path in list(self.data_dir.glob(f"**/*.{ss_file_ext}")):
                seq, _ = parse_sec_struct_file(ss_file_path)

                if len(seq) >= min_seq_len and len(seq) <= max_seq_len:
                    self.ss_paths.append(ss_file_path)

    def __len__(self):
        return len(self.ss_paths)

    def __getitem__(self, idx):
        ss_id = self.ss_paths[idx].stem
        seq, sec_struct = parse_sec_struct_file(self.ss_paths[idx])

        #seq_encoded = torch.tensor(self.alphabet.encode(seq), dtype=torch.int64)
        chars = '<s> ' + ' '.join(list(seq))
        seq_encoded = task.source_dictionary.encode_line( chars,  add_if_not_exist=False )
        sec_struct = torch.tensor(sec_struct)
        #return ss_id, seq, seq_encoded, sec_struct
        #return  seq, seq_encoded, sec_struct
        return  seq, seq_encoded, sec_struct, ss_id


def my_collate(batch):
    tokens = [ torch.Tensor(item[ 'token']) for item in batch]
    target = torch.stack([ torch.tensor(item['label']) for item in batch])
    pad_tokens = collate_tokens(
        tokens,
        pad_idx=task.source_dictionary.pad(),
        eos_idx=task.source_dictionary.eos(),
        left_pad=False,pad_to_length=None,pad_to_bsz=None
    )
    pad_tokens = pad_tokens.long()
    return [ pad_tokens, target]


def acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs):
    acc = accuracy_score(labels, preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs)
    aupr = average_precision_score(labels, probs)
    return {
        "acc": acc,
        "auc": auc,
        "aupr": aupr,
        "f1": f1,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
    }


eval_summary = [] #cell, fold, acc, auc, auprc, f1, mcc, prec, recall



class hybridSecStruct(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(hybridSecStruct, self).__init__()
        #self.fc2 = nn.Linear( input_dim, output_dim)
        self.dropout = nn.Dropout( 0.1)
        self.threshold = 0.1

        self.hybrid = models[0].encoder
        self.ss_head = SecStructPredictionHead( input_dim, 12 ) 

    def forward(self, src_tokens): # input token ids
        #print( src_tokens.shape) #  B x L
        x = self.hybrid( src_tokens, features_only=True)[0]
        x = self.dropout(x)
        logits = self.ss_head(x[..., 1:-1,:] ).squeeze(-1)
        return logits



scaler = GradScaler()

#print(model)


pos_weight = torch.Tensor([10]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

learning_rate = 0.0001


def contact2ct(contact, seq):
    seq_len = len(seq)
    structure = np.where(contact)
    pair_dict = dict()
    for i in range(seq_len):
        pair_dict[i] = -1
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]
    first_col = list(range(1, seq_len+1))
    second_col = list(seq)
    third_col = list(range(seq_len))
    fourth_col = list(range(2, seq_len+2))
    fifth_col = [pair_dict[i]+1 for i in range(seq_len)]
    last_col = list(range(1, seq_len+1))
    df = pd.DataFrame()
    df['index'] = first_col
    df['base'] = second_col
    df['index-1'] = third_col
    df['index+1'] = fourth_col
    df['pair_index'] = fifth_col
    df['n'] = last_col
    return df


L_train = []
L_val =[]
AUC = []
PRAUC =[]
testSPR = []


test_list = ['bpRNA']


for celldir in test_list:
    celltype= celldir
    ds_test = SecondaryStructureDataset('../data/bpRNA/test') 
    print(celltype)
    
    #valid_loader = DataLoader(ds_valid, shuffle=False, batch_size=1)
    test_loader = DataLoader(ds_test, shuffle=False, batch_size=1)

    models, _model_args = checkpoint_utils.load_model_ensemble([path], task=task)
    model =  hybridSecStruct( input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load("../weights/HydraRNA_SS_model.pt"))
 
    model.to(device)
    print(model)
    num_params = sum(param.numel() for param in model.parameters())
    print('Number of parameters: %d'%(num_params) )


    bestbest_threshold = 0.5
    bestbest_val = 0
    num_epochs = 1
    for epoch in tqdm(range(num_epochs)):

        counter = 0
        seq_list = []
        fpath_list = []
        test_f1 = []
        test_precision = []
        test_recall = []
        current_test_loss = 0.0
        with torch.no_grad():
            for seq, data, label,ss_id in tqdm(test_loader):  # seq is a list of sequence, with len==batch_size==1
                seq_list.append(seq[0])
                fpath_list.append( ss_id[0] )
                #print( ss_id[0])
                counter += 1
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(data.to(device) ).float()
                    seq_len = label.shape[1]
                    upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=logits.device), diagonal=1)
                    loss = criterion(logits[..., upper_tri_mask], label.to(device)[..., upper_tri_mask])
                current_test_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                #print(probs.shape, len(seq))
                #print (probs.min(), probs.mean(), probs.max())
                sec_struct_pred = prob_mat_to_sec_struct(probs=probs[0], seq=seq[0], threshold= bestbest_threshold)
                f1 = ss_f1( label[0], sec_struct_pred, allow_flexible_pairings=False)
                test_f1.append( f1 )
                prec = ss_precision( label[0], sec_struct_pred, allow_flexible_pairings=False)
                test_precision.append( prec )
                reca = ss_recall( label[0], sec_struct_pred, allow_flexible_pairings=False)
                test_recall.append( reca )
                #print( loss.item(), f1 )
                #if counter > 20:
                #    break
                ct = contact2ct(sec_struct_pred, seq[0])
                #print(ct)
                ct.to_csv('predict/predict_HydraRNA_%s.ct'%(ss_id[0]),sep='\t',index=None,header=None)

            curr_metric_test = sum(test_f1)/len(test_f1)
            curr_prec_test = sum(test_precision)/len(test_precision)
            curr_reca_test = sum(test_recall)/len(test_recall)
            test_loss = current_test_loss / counter

        print("loss_Test: ", test_loss, "F1_Test: %.4f"%curr_metric_test, "Precision_Test: %.4f"%curr_prec_test, "Recall_Test: %.4f"%curr_reca_test)
        sys.stdout.flush()
        with open('bpRNA_test_HydraRNA_predict_resulst_NOTallow_flexible_pairings.csv','w') as fout:
            fout.write( 'seq\tF1\tPrec\tRecall\tfile\n')
            for x,y,a,b,c in zip(seq_list, test_f1, test_precision, test_recall, fpath_list):
                fout.write( x+'\t'+str(y)+ '\t'+str(a)+ '\t'+str(b)+ '\t'+ c +'\n')
    
      

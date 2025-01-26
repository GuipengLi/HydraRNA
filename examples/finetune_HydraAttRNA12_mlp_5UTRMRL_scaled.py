import pandas as pd
import numpy as np

import os
import sys
from pathlib import Path
from sklearn import metrics

from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm


from sklearn.metrics import precision_recall_curve, auc, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import spearmanr, pearsonr

from sklearn import preprocessing

from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data.data_utils import collate_tokens

from torch.amp import GradScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path='../weights/HydraRNA_model.pt' # change the path if necessary

parser = options.get_generation_parser(default_task='masked_lm_span')
args = options.parse_args_and_arch(parser,  ['../dict/'])

# Setup task
task = tasks.setup_task(args)
print('| loading model from {}'.format(path))
models, _model_args = checkpoint_utils.load_model_ensemble([path], task=task)

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

input_dim = 1024
hidden_dim = 40
output_dim = 1

num_epochs = 50

batch_size = 32


MODEL = 'finetune5UTRMRL_HydraAttRNA12_mlp'

class FastaDataset(Dataset):
    def __init__(self, csvfile):
        self.fadf = pd.read_csv( csvfile , sep=',')

    def __len__(self):
        return len(self.fadf)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.fadf.utr[idx] # utr
        if 'scaled_rl' in self.fadf.columns:
            cls = self.fadf.scaled_rl[idx]
        else:
            cls = self.fadf.rl[idx]
        
        #if len(seq) > 4000:
        #    seq = seq[0:4000]
        chars = '<s> ' + ' '.join(list(seq))
        tokens = task.source_dictionary.encode_line(
                chars, add_if_not_exist=False )

        sample = {'token':tokens, 'label':cls }
        return sample

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

ds_train = FastaDataset( '../data/e_train.csv')
ds_valid = FastaDataset( '../data/e_test.csv')
ds_test = FastaDataset( '../data/subhuman.csv')

train_loader = DataLoader(ds_train, shuffle=True, batch_size=batch_size, collate_fn=my_collate)
valid_loader = DataLoader(ds_valid, shuffle=False, batch_size=batch_size, collate_fn=my_collate)
test_loader = DataLoader(ds_test, shuffle=False, batch_size=batch_size, collate_fn=my_collate)



class hybridMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(hybridMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout( 0.1)

        self.hybrid = models[0].encoder


    def forward(self, src_tokens): # input token ids
        #print( src_tokens.shape) #  B x L
        x = self.hybrid( src_tokens, features_only=True)[0]
        pad_mask = src_tokens.eq(1)
        has_pads = pad_mask.any()
        tmp = 1 - pad_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        denom = torch.sum( tmp[:,1:-1,:], 1 )

        x = x * tmp

        x = x[:,1:-1,:].sum(dim=1)/denom  # from 99 x 132 x 1024 to 99 x 1024
        #x = x[:,1:-1,:].mean(dim=1)  # from 99 x 132 x 1024 to 99 x 1024
        #x = x[:,0,:]
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model =  hybridMLP( input_dim, hidden_dim, output_dim)

model.to(device)
scaler = GradScaler()


print(model)

num_params = sum(param.numel() for param in model.parameters())

print('Number of parameters: %d'%(num_params) )



criterion = nn.MSELoss()


learning_rate = 0.00005


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate )


min_validation_loss = 9999
L_train = []
L_val =[]
AUC = []
PRAUC =[]
testSPR = []

tdf = pd.read_csv('../data/e_train.csv')
stdscaler = preprocessing.StandardScaler()
stdscaler.fit( np.array(tdf.rl).reshape((-1,1)) )



for epoch in tqdm(range(num_epochs)):

    train_running_loss = 0.0
    counter = 0
    model.train()
    for data,label in train_loader:
        optimizer.zero_grad()
        counter += 1
        #print( data.shape)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(data.to(device) ).float()

            loss = criterion(output, label.unsqueeze(1).float().to(device))
        #print( loss.item())
        train_running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    TL = train_running_loss / counter
    L_train.append(TL)
    model.eval()
    PREDICT = []
    LABEL = []
    counter = 0
    with torch.no_grad():
        current_valid_loss = 0.0
        for SEQ, Z in valid_loader:
            counter += 1
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(SEQ.to(device) ).float()
                loss = criterion(output, Z.unsqueeze(1).float().to(device))
            current_valid_loss += loss.item()
            PREDICT.extend(output.cpu().numpy())
            LABEL.extend(Z.cpu().numpy())
        T_loss = current_valid_loss / counter
        L_val.append(T_loss)


        PP = np.array(PREDICT)
        TT = np.array(LABEL)
        flattened_array1 = PP.flatten()
        flattened_array2 = TT.flatten()
        flattened_array1 = stdscaler.inverse_transform(flattened_array1.reshape((-1,1)) )
        flattened_array1 = flattened_array1.flatten()

        corr = spearmanr(flattened_array1, flattened_array2)[0]
        pcorr = pearsonr(flattened_array1, flattened_array2)[0]
        mse = mean_squared_error(flattened_array2, flattened_array1)
        #corr = res.statistic
        if min_validation_loss > mse:
            min_validation_loss = mse
            best_epoch = epoch
            print('Min val loss ' + str(min_validation_loss) + ' in epoch ' + str(best_epoch))
            torch.save(model.state_dict(), fr"./model_{MODEL}_valid.pt")

        PRAUC.append(corr)

    PREDICT =[]
    LABEL=[]
    with torch.no_grad():
        for SEQ, Z in test_loader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(SEQ.to(device) ).float()
            PREDICT.extend(output.cpu().numpy())
            LABEL.extend(Z.cpu().numpy())
    PP = np.array(PREDICT)
    TT = np.array(LABEL)
    flattened_array1 = PP.flatten()
    flattened_array2 = TT.flatten()
    flattened_array1 = stdscaler.inverse_transform(flattened_array1.reshape((-1,1)) )
    flattened_array1 = flattened_array1.flatten()

    pcorr2 = pearsonr(flattened_array1, flattened_array2)[0]
    corr2 = spearmanr(flattened_array1, flattened_array2)[0]
    mse2 = mean_squared_error(flattened_array2, flattened_array1)
    testSPR.append( corr2)

    print("Train loss: ", TL, "mse_valid: ", mse, 'cor_valid: %.4f'%corr,'pcor_valid: %.4f'%pcorr, 'mse_test: %.4f'%mse2,  'cor_test: %.4f'%corr2, 'pcor_test: %.4f'%pcorr2)
    sys.stdout.flush()



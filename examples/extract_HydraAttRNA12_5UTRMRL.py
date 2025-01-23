import sys
import torch
import os
import inspect
from fairseq import checkpoint_utils, data, options, tasks
import pandas as pd
from tqdm import tqdm
import pysam

#from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset, Dataset


path='../weights/HydraRNA_model.pt' # change the path if necessary

parser = options.get_generation_parser(default_task='masked_lm_span')
args = options.parse_args_and_arch(parser,  ['../dict/'])
# Setup task
task = tasks.setup_task(args)
print('| loading model from {}'.format(path))
models, _model_args = checkpoint_utils.load_model_ensemble([path], task=task)
model = models[0]


model.to('cuda')
model.half()


def parse_csv( csvfile):
    DATA = []
    LABEL = []
    df =  pd.read_csv( csvfile , sep=',')

    for record in tqdm(df.itertuples(), desc="Processing records", unit="record"):
        seq = record.utr100.strip()

        chunk_size = 10240
        seq_chunks = [ seq[i-chunk_size:i] for i in range( chunk_size, len(seq)+chunk_size, chunk_size)]
        cls = record.rl
        #print (cls, str2id(cls) ,seq)
        seq_chunks = ['<s> ' + ' '.join(list(x)) for x in seq_chunks ]
        tokens_chunks = [ task.source_dictionary.encode_line(
            chars, add_if_not_exist=False ) for chars in seq_chunks]

        batch = data.monolingual_dataset.collate(
            samples=[{'id': -1, 'source': tokens, 'target':tokens} for tokens in tokens_chunks],  # bsz = chunk_size
            pad_idx=task.source_dictionary.pad(),
            eos_idx=task.source_dictionary.eos(),
        )
        xx = batch['net_input']
        model.eval()
        with torch.no_grad():
            y=model.encoder.extract_features( src_tokens= xx['src_tokens'].to('cuda') )
            mean_tensor = torch.mean(y[0][:,1:-1,:], dim=1 ).mean(dim=0, keepdim=True)
            DATA.append(mean_tensor)
            label_tensor = torch.tensor( cls )
            LABEL.append(label_tensor)

    batched_DATA = torch.cat(DATA, dim=0)
    batched_LABEL = torch.stack(LABEL)
    print(batched_DATA.shape)
    print(batched_LABEL.shape)
    dataset = TensorDataset( batched_DATA, batched_LABEL)
    return( dataset)
    

MODEL = 'HydraRNA_5UTRMRL'

dataset1 = parse_csv( '../data/e_train.csv')
fout = 'data_features_%s_train.pt'%MODEL
print('saving to %s'%fout)
torch.save ( dataset1, fout)

dataset1 = parse_csv( '../data/subhuman.csv')
fout = 'data_features_%s_valid.pt'%MODEL
print('saving to %s'%fout)
torch.save ( dataset1, fout)

dataset1 = parse_csv( '../data/e_test.csv')
fout = 'data_features_%s_test.pt'%MODEL
print('saving to %s'%fout)
torch.save ( dataset1, fout)

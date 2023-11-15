import os
import tempfile
 
from primeqa.ir.dense.colbert_top.colbert.utils.utils import create_directory, print_message
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.training.training import train
from primeqa.ir.dense.colbert_top.colbert.indexing.collection_indexer import encode
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher
import pandas as pd
import numpy as np
import torch
from IPython.display import display, HTML
from datasets import load_dataset,  concatenate_datasets

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

random_seed = 2
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


# path for index file
collection_fn = '/raid/nlp/sameer/NL2bash/colbert_tsv_files/index_descriptions_examples.tsv'

args_dict = {
                'root': os.path.join("/tmp/tmphdtvb4y3/output_dir",'test_indexing'),
                'experiment': 'test_indexing',
                'checkpoint': "/tmp/tmphdtvb4y3/output_dir/test_training/2023-11/15/13.34.02/checkpoints/colbert-LAST.dnn",
                'collection': collection_fn,
                'index_root': os.path.join("/tmp/tmphdtvb4y3/output_dir", 'test_indexing', 'indexes'),
                'index_name': 'index_name',
                'model_type': 'roberta-base',
                'nranks': 1,
                'amp' : True,
            }

# run indexing step
with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
    colBERTConfig = ColBERTConfig(**args_dict)
    create_directory(colBERTConfig.index_path_)
    encode(colBERTConfig, collection_fn, None, None)

# path for queries (retrieval step)
queries_fn = "/raid/nlp/sameer/NL2bash/colbert_tsv_files/queries.tsv"

args_dict = {
                'root': "/tmp/tmphdtvb4y3/output_dir",
                'experiment': 'test_indexing' ,
                'checkpoint': "/tmp/tmphdtvb4y3/output_dir/test_training/2023-11/15/13.34.02/checkpoints/colbert-LAST.dnn",
                'model_type': 'roberta-base',
                'index_location': os.path.join("/tmp/tmphdtvb4y3/output_dir", 'test_indexing', 'indexes', 'index_name'),
                'queries': queries_fn,
                'bsize': 1,
                'topK': 5, # retrieve top 5 hits
                'rank': 0,
                'nranks': 1,
                'amp': True,
            }

with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
    colBERTConfig = ColBERTConfig(**args_dict)
    searcher = Searcher(args_dict['index_location'], checkpoint=args_dict['checkpoint'], config=colBERTConfig)
    rankings = searcher.search_all(args_dict['queries'], args_dict['topK'])

print(rankings.flat_ranking[:10])

with open(queries_fn, 'r') as f:
    for line in f.readlines():
        if str(rankings.flat_ranking[0][0]) == line.split()[0]:
            print(line)

with open(collection_fn, 'r') as f:
    for line in f.readlines():
        if str(rankings.flat_ranking[0][1]) == line.split()[0]:
            print(line)

"""Below steps are for evaluation or calculating accuracy of top 5 retrieval. 
All the retrieval results are stored in rankings.flat_ranking as a list of tuples. Each tuple has following format
(query-id, retrived doc-index, rank, score)"""

# store the mapping of index to text
f = open(collection_fn, 'r')
descriptions = f.readlines()

index2text = {}
for line in descriptions:
    index2text[line.split("\t")[0]] = line.split("\t")[2]

# for each query store title of top 5 retrieved documents
mapping = {}
for tup in rankings.flat_ranking:
    if tup[0] not in mapping:
        mapping[tup[0]] =  []
    if tup[1] == '0':
        continue
    mapping[tup[0]].append(index2text[str(tup[1])].strip())   # here error occurs. as tuple tup consist of retrived document with index 0. But no such document exists.

tldr = load_dataset("neulab/tldr")
tldr_train_test = concatenate_datasets([tldr['train'], tldr['test']])
ds = tldr_train_test.train_test_split(test_size=0.15, seed=4)
ds["test"] = ds["test"].filter(lambda example: example["cmd_name"] in ds["train"]["cmd_name"])
print("length of train set is: ", len(ds["train"]))
print("length of test set: ", len(ds["test"]))

tldr_train, tldr_val = ds["train"], ds["test"]

cnt = 0
for i in range(0,len(tldr_val)):
    if tldr_val[i]["cmd_name"] in mapping[str(i+1)]:
        cnt += 1

print(cnt/len(tldr_val))
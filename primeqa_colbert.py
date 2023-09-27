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

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

random_seed = 2 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


text_triples_fn  = '/home1/tejomay/NL2bash/train_sbert_triples.tsv'
model_type = 'roberta-base'
with tempfile.TemporaryDirectory() as working_dir:
    output_dir=os.path.join(working_dir, 'output_dir')

data = pd.read_csv(text_triples_fn, sep='\t', header=None)

print

data = data.drop([0], axis=1)
data.rename(columns = {1:0, 2:1, 3:2}, inplace = True)

print("printing output dir: ",output_dir)
args_dict = {
                'root': output_dir,
                'experiment': 'test_training',
                'triples': text_triples_fn,
                'model_type': model_type,
                'bsize': 16,
                'epochs': 3,
                'nranks': 1,
                'amp' : True,
                # 'lr' : 3e-5
            }

with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
    colBERTConfig = ColBERTConfig(**args_dict)
    latest_model_fn = train(colBERTConfig, text_triples_fn, None, None)

collection_fn = '/home1/tejomay/NL2bash/index_descriptions.tsv'

data = pd.read_csv(collection_fn, sep='\t')

print(data.head(3))
print("---------------------------")
args_dict = {
                'root': os.path.join(output_dir,'test_indexing'),
                'experiment': 'test_indexing',
                'checkpoint': latest_model_fn,
                'collection': collection_fn,
                'index_root': os.path.join(output_dir, 'test_indexing', 'indexes'),
                'index_name': 'index_name',
                'doc_maxlen': 180,
                'num_partitions_max': 2,
                'kmeans_niters': 1,
                'nway': 1,
                'rank': 0,
                'nranks': 1,
                'amp' : True,
            }

with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
    colBERTConfig = ColBERTConfig(**args_dict)
    create_directory(colBERTConfig.index_path_)
    encode(colBERTConfig, collection_fn, None, None)

queries_fn = "/home1/tejomay/NL2bash/queries.tsv"

args_dict = {
                'root': output_dir,
                'experiment': 'test_indexing' ,
                'checkpoint': latest_model_fn,
                'model_type': model_type,
                'index_location': os.path.join(output_dir, 'test_indexing', 'indexes', 'index_name'),
                'queries': queries_fn,
                'bsize': 1,
                'topK': 3,
                'nway': 1,
                'rank': 0,
                'nranks': 1,
                'amp': True,
            }

with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
    colBERTConfig = ColBERTConfig(**args_dict)
    searcher = Searcher(args_dict['index_location'], checkpoint=args_dict['checkpoint'], config=colBERTConfig)
    rankings = searcher.search_all(args_dict['queries'], args_dict['topK'])

rankings.flat_ranking[0]

with open(queries_fn, 'r') as f:
    for line in f.readlines():
        if str(rankings.flat_ranking[0][0]) == line.split()[0]:
            print(line)

with open(collection_fn, 'r') as f:
    for line in f.readlines():
        if str(rankings.flat_ranking[0][1]) == line.split()[0]:
            print(line)
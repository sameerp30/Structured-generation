from transformers import AutoTokenizer, BertModel
import torch
from datasets import load_dataset, concatenate_datasets
import json
import numpy as np
from numpy.linalg import norm
import operator
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import sys

num_negatives = sys.argv[1] if len(sys.argv)>=2 else 10
random_seed = 2
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

json_file_path = "/raid/nlp/sameer/ColBERT/descritions_80.json"  # enter the path for cmd name to description file

embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')

# tldr = load_dataset("neulab/tldr")
# tldr_train_test = concatenate_datasets([tldr['train'], tldr['test']])
# ds = tldr_train_test.train_test_split(test_size=0.15, seed=4)
# ds["test"] = ds["test"].filter(lambda example: example["cmd_name"] in ds["train"]["cmd_name"])
# print("length of train set is: ", len(ds["train"]))
# print("length of test set: ", len(ds["test"]))

# tldr_train, tldr_val = ds["train"], ds["test"]

with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())

contents = {i:contents[i] for i in contents.keys() if contents[i]!=""}

print("length of contents after: ", len(contents))

contents2cmd = {}

for cmd in contents:
    contents2cmd[contents[cmd]] = cmd

corpus = list(contents.values())
# cmd_names = list(contents.values())

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


most_similar = {}

desc_embeddings = {}
top_k = min(num_negatives + 1, len(corpus))

for i,x in enumerate(tqdm(tldr_train)):  # Loop over all nl queries tldr_train["cmd_name"][i] is command name for ith query and tldr_train["nl"][i] is natural language query for ith sample
    query = tldr_train["nl"][i]     
    results = []
    if tldr_train["cmd_name"][i] not in contents.keys():
        continue
    query_embedding = embedder.encode(contents[tldr_train["cmd_name"][i]], convert_to_tensor=True)  # tldr_train["cmd_name"][i] is reference module name for corresponding nl_query
    query_embedding = query_embedding.to("cuda:0")
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k, largest=True, sorted=True)
    
    for idx in top_results[1]:
        results.append(contents2cmd[corpus[idx]])

    if tldr_train["cmd_name"][i] in results[:num_negatives+1]:
        results.remove(tldr_train["cmd_name"][i])
    most_similar[tldr_train["nl"][i]] = results[:num_negatives + 1]

with open("most_3_Sbert_similar_train.json", "w") as outfile:
    json.dump(most_similar, outfile)
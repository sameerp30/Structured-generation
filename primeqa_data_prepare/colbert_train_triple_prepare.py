import pandas as pd
import requests
from bs4 import BeautifulSoup
from rank_bm25 import *
from rank_bm25 import BM25Okapi
import json
from datasets import load_dataset


json_file_path_1 = "/home1/tejomay/NL2bash/train_descriptions.json"
json_file_path_2 = "/home1/tejomay/NL2bash/least_2_Sbert_similar_train.json"
query_positive = {}
query_negative = {}
query_all = []
positive_passage_all = []
negative_passage_all = []

with open(json_file_path_1, 'r') as j:
    contents = json.loads(j.read())

with open(json_file_path_2, 'r') as j:
    query_negative_index = json.loads(j.read())

for key in query_negative_index:
    query_negative[key] = [contents[i] for i in query_negative_index[key]]


tldr = load_dataset("neulab/tldr")

keys_with_desc = contents.keys()
common_keys = set(keys_with_desc).intersection(set(tldr["train"]["cmd_name"]))


contents = {i: contents[i] for i in contents.keys() if contents[i]!=""}
descriptions = list(contents.values())

query_negative_filter = {}
for i in range(0,len(tldr["train"])):
    cmd_name = tldr["train"]["cmd_name"][i]
    if cmd_name not in contents:
        continue
    query_positive[tldr["train"]["nl"][i]] = contents[cmd_name]
    # query_negative[tldr["train"]["nl"][i]] = list(set(descriptions).difference(set(contents[cmd_name])))
    query_negative_filter[tldr["train"]["nl"][i]] = query_negative[tldr["train"]["nl"][i]]
    # query_negative[tldr["train"]["nl"][i]] = descriptions_temp

print(len(query_positive))
print(len(query_negative_filter))
for query in query_negative_filter:
    for i in range(0,len(query_negative_filter[query])):
        query_all.append(query)
        positive_passage_all.append(query_positive[query])
        negative_passage_all.append(query_negative_filter[query][i])

d = {"query": query_all, "positive_passage": positive_passage_all, "negative_passage": negative_passage_all}
df = pd.DataFrame(d)
df.to_csv('train_sbert_triples.tsv', sep="\t",index=False)

print(positive_passage_all[0])
print("--------------------------------------------------------------------------------------------")
print(positive_passage_all[1])
print("-------------------------")
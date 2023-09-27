import pandas as pd
from datasets import load_dataset
import json

data = pd.read_csv("/home1/tejomay/NL2bash/train_sbert_triples.tsv", sep='\t', header=None)
# print(data.head())
json_file_path = "/home1/tejomay/NL2bash/train_descriptions.json"


with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())

contents = {i:contents[i] for i in contents.keys() if contents[i]!=""}
cnt = 1
text = []
title = []
id = []
for cmd in contents:
    text.append(contents[cmd])
    title.append(cmd)
    id.append(cnt)
    cnt += 1

'''desc_dict = {}
for i in range(0,len(data[2])):
    if data[2][i] not in desc_dict:
        desc_dict[data[2][i]] = 1

description = list(desc_dict.keys())
id = []
for i in range(1,len(description)+1):
    id.append(i)

title = []
for i in range(1,len(id)+1):
    title.append("description" + "_" + str(i))'''


details = {"id": id, "text": text, "title": title}
df = pd.DataFrame(details)
df.reset_index(drop=True, inplace=True)

print(df.head())

df.to_csv('index_descriptions.tsv', sep="\t",index=False)
tldr = load_dataset("neulab/tldr")

id = []
queries = []
for i in range(0, 1):
    id.append(i+1)
    queries.append(tldr["train"]["nl"][i])

df = pd.DataFrame([*zip(id,queries)])
df.to_csv('queries.tsv', sep="\t", index=False)

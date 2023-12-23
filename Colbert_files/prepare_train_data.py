from random import sample
import json
import pandas as pd
import re

data = pd.read_csv("/raid/nlp/sameer/NL2bash/train_Sbert_triples_1_pos_3_neg.tsv", sep='\t', header=None)
queries_data = pd.read_csv("/raid/nlp/sameer/ColBERT/colbert_tsv_files/queries_train.tsv", sep='\t', header=None)

# print(data.head())
json_file_path = "/raid/nlp/sameer/ColBERT/descritions_80.json"

def remove_punct(test_str):
    punc = '''!()[]{};:'"\,<>./?@#$%^&*~'''
 
# Removing punctuations in string
# Using loop + punctuation string
    for ele in test_str:
        if ele in punc:
            test_str = test_str.replace(ele, " ")
    return test_str

with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())

for x in contents:
    contents[x] = remove_punct(contents[x])
    contents[x] = re.sub(r'\s+', ' ', contents[x])
    contents[x] = contents[x].strip()

cnt = 0
desc2id = {}
for cmd in contents:
    desc2id[contents[cmd]] = cnt
    cnt += 1

print(queries_data.columns)
query2id = {}
for i, row in queries_data.iterrows():
    query2id[row[1]] = row[0]

# print(data.head(3))
with open('colbert_triples_1_pos_3_neg.json', 'w') as triples_file:
    for i,row in data.iterrows():
        temp = []
        triples_file.write(json.dumps([query2id[row[0]], desc2id[row[1]], desc2id[row[2]]])  + '\n')
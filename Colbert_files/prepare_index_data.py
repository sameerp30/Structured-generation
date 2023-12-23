import pandas as pd
from datasets import load_dataset, concatenate_datasets
import json
import re
import string

def remove_punct(test_str):
    punc = '''!()[]{};:'"\,<>./?@#$%^&*~'''
 
# Removing punctuations in string
# Using loop + punctuation string
    for ele in test_str:
        if ele in punc:
            test_str = test_str.replace(ele, " ")
    return test_str

json_file_path = "/raid/nlp/sameer/ColBERT/descritions_80.json"  # path for cmd name to descriptions file


with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())

print("length of contents before: ", len(contents))
contents = {i:contents[i] for i in contents.keys() if contents[i]!=""}

print("length of contents after: ", len(contents))
for x in contents:
    contents[x] = remove_punct(contents[x])
    contents[x] = re.sub(r'\s+', ' ', contents[x])
    contents[x] = contents[x].strip()


cnt = 0
text = []
title = []
id = []
for cmd in contents:
    text.append(contents[cmd])
    title.append(cmd)
    id.append(cnt)
    cnt += 1


details = {"id": id, "text": text}
df = pd.DataFrame(details)
df.reset_index(drop=True, inplace=True)
df.to_csv('/raid/nlp/sameer/ColBERT/colbert_tsv_files/index.tsv', sep="\t",index=False, header=False)

details = {"id": id, "text": text, "title": title}
df = pd.DataFrame(details)
df.reset_index(drop=True, inplace=True)

df.to_csv('/raid/nlp/sameer/ColBERT/colbert_tsv_files/index_with_title.tsv', sep="\t",index=False, header=False)


# tldr = load_dataset("neulab/tldr")
# tldr_train_test = concatenate_datasets([tldr['train'], tldr['test']])
# ds = tldr_train_test.train_test_split(test_size=0.15, seed=4)
# ds["test"] = ds["test"].filter(lambda example: example["cmd_name"] in ds["train"]["cmd_name"])
# print("length of train set is: ", len(ds["train"]))
# print("length of test set: ", len(ds["test"]))

# tldr_train, tldr_val = ds["train"], ds["test"]

id = []
queries = []
for i in range(0, len(tldr_train)):     # loop over all training nl query samples
    id.append(i)
    queries.append(tldr_train["nl"][i])

details = {"id": id, "text": queries}
df = pd.DataFrame(details)
df.reset_index(drop=True, inplace=True)

print(df.head())

df.to_csv('/raid/nlp/sameer/ColBERT/colbert_tsv_files/queries_train.tsv', sep="\t",index=False, header=False)
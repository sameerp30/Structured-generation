from transformers import AutoTokenizer, BertModel
import torch
from datasets import load_dataset
import json
import numpy as np
from numpy.linalg import norm
import operator
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


random_seed = 2
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

model.to(device)
json_file_path = "/home1/tejomay/NL2bash/train_descriptions.json"

embedder = SentenceTransformer('all-MiniLM-L6-v2')


with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())

contents = {i:contents[i] for i in contents.keys() if contents[i]!=""}

contents2cmd = {}

for cmd in contents:
    contents2cmd[contents[cmd]] = cmd

corpus = list(contents.values())
cmd_names = list(contents.values())

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
print(type(corpus_embeddings))
tldr = load_dataset("neulab/tldr")


least_similar = {}

desc_embeddings = {}
top_k = min(3, len(corpus))
for i,x in enumerate(tqdm(tldr["train"]["nl"])):
    query = tldr["train"]["nl"][i]
    results = []
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    # We use cosine-similarity and torch.topk to find the highest 5 scores
    # corpus_temp = list(contents_temp.values())
    # corpus_embeddings = embedder.encode(corpus_temp, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k, largest=True, sorted=True)
    
    for idx in top_results[1]:
        results.append(contents2cmd[corpus[idx]])

    if tldr["train"]["cmd_name"][i] in results[:2]:
        results.remove(tldr["train"]["cmd_name"][i])
    least_similar[tldr["train"]["nl"][i]] = results[:2]

# for i,x in enumerate(tqdm(tldr["train"])):
#     inputs = tokenizer(tldr["train"]["nl"][i], return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
#     outputs = model(**inputs)
#     last_hidden_states = outputs.last_hidden_state
#     nl_vector = return_vetor(last_hidden_states[0])
#     score = {}
#     for key in contents:
#         score[key] = cosine_similarity(nl_vector, desc_embeddings[key])
#     score = dict( sorted(score.items(), key=operator.itemgetter(1),reverse=True))
#     # keys_desc_order = [i.split("_")[0] for i in list(score.keys())]
#     last_3 = list(score.keys())[-3:]
#     least_similar[tldr["train"]["nl"][i]] = last_3

with open("least_2_Sbert_similar_train.json", "w") as outfile:
    json.dump(least_similar, outfile)


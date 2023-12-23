from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from utils.util import dedup_results, get_bag_of_keywords, clean_anonymize_command, get_bag_of_words
from collections import defaultdict, Counter
import numpy as np
from datasets import load_dataset, concatenate_datasets


device = 'cuda:0'
# base_model_name = 'neulab/docprompting-tldr-gpt-neo-1.3B'
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# model = AutoModelForCausalLM.from_pretrained(base_model_name)
# model = model.to(device)

with open('/raid/nlp/sameer/docprompting/codet5_t10.json', 'r') as f:
    examples = json.load(f)

with open('/raid/nlp/sameer/docprompting/alternate_flag_info.json', 'r') as f:
    alternate_flag_info = json.load(f)

tldr = load_dataset("neulab/tldr")
tldr_train_test = concatenate_datasets([tldr['train'], tldr['test']])
ds = tldr_train_test.train_test_split(test_size=0.15, seed=4)
test_set = ds["test"].filter(lambda example: example["cmd_name"] in ds["train"]["cmd_name"])

def token_prf(tok_gold, tok_pred, pred_cmd, gt_cmd, match_blank=False):
    if match_blank and len(tok_gold) == 0: # do not generate anything
        if len(tok_pred) == 0:
            m = {'r': 1, 'p': 1, 'f1': 1}
        else:
            m = {'r': 0, 'p': 0, 'f1': 0}
    else:
        tok_gold_dict = Counter(tok_gold)
        tok_pred_dict = Counter(tok_pred)
        tokens = set([*tok_gold_dict] + [*tok_pred_dict])
        hit = 0
        
        for token in tokens:
            if token[0]=="-" and pred_cmd + ".txt" in alternate_flag_info and token in alternate_flag_info[pred_cmd + ".txt"]:
                hit += min( max([tok_pred_dict.get(flag, 0) for flag in [token] + alternate_flag_info[pred_cmd + ".txt"][token]["anyof"]]), tok_gold_dict.get(token, 0))
            else:
                hit += min(tok_gold_dict.get(token, 0), tok_pred_dict.get(token, 0))

        p = hit / (sum(tok_pred_dict.values()) + 1e-10)
        r = hit / (sum(tok_gold_dict.values()) + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        m = {'r': r, 'p': p, 'f1': f1}
    return m
    
def measure_bag_of_word(gold, pred, pred_cmd, gt_cmd):
    tok_gold = get_bag_of_words(gold)
    tok_pred = get_bag_of_words(pred)
    tok_pred_guidance = get_bag_of_words(pred_guidance)

    m = token_prf(tok_gold, tok_pred) # whole sentence
    # print("printing tok gold: ",tok_gold)
    # print("printing tok pred: ",tok_pred)
    gold_cmd = tok_gold[0] if len(tok_gold) else "NONE_GOLD"
    pred_cmd = tok_pred[0] if len(tok_pred) else "NONE_PRED"

    prompt_cmd_name = [gt_cmd]
    if "-" in prompt_cmd_name[0]:
        prompt_cmd_name.append(prompt_cmd_name[0][:prompt_cmd_name[0].find("-")] + " " + prompt_cmd_name[0][prompt_cmd_name[0].find("-")+1:])
        if prompt_cmd_name[0].count("-") > 1:
                prompt_cmd_name.append(" ".join(prompt_cmd_name[0].split("-")))

    m = {**m, 'cmd_acc': max([int(k in pred) for k in prompt_cmd_name])}   # command accuracy

    no_cmd_m = token_prf(tok_gold[1:], tok_pred[1:], pred_cmd, gt_cmd, match_blank=True)
    no_cmd_m = {f"no_cmd_{k}": v for k, v in no_cmd_m.items()}
    m = {**m, **no_cmd_m}
    return m
    
f = open("prediction_base_greedy.txt", "r")
pred_list = [m.strip() for m in f.readlines()]
f = open("pred_cmd.txt", "r")
pred_cmd_list = [m.strip() for m in f.readlines()]

f = open("source.txt","r")
src_list = [m.strip() for m in f.readlines()]

metric_list = defaultdict(list)
cnt = 0
gt_cmd_list = list(test_set["cmd_name"]) # gold command name
for src, pred, pred_cmd, gt_cmd in zip(src_list, pred_list, pred_cmd_list, gt_cmd_list):
    for k, v in measure_bag_of_word(src, pred, gt_cmd).items():
            metric_list[k].append(v)
for k, v in metric_list.items():
    metric_list[k] = np.mean(v)
print(metric_list)
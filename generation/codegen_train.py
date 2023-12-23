from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import Trainer, TrainingArguments
import json
from transformers import default_data_collator
from huggingface_hub import login
import transformers
import torch


login(token="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")


tldr = load_dataset("neulab/tldr")

json_file_path = "/raid/nlp/sameer/NL2bash/flags.json"
context_file_path = "/raid/nlp/sameer/NL2bash/docprompting_data/tldr/manual_section.json"


with open(json_file_path, 'r') as j:
    flags_file = json.loads(j.read())

with open(context_file_path, 'r') as j:
    context_file = json.loads(j.read())

# torch.cuda.set_device(1)
cmd_flags = [i.split(".")[0] for i in flags_file.keys()]
print(cmd_flags[:10])

def prepare_data(context=False):
    tldr = load_dataset("neulab/tldr")
    tldr_train_test = concatenate_datasets([tldr['train'], tldr['test']])
    ds = tldr_train_test.train_test_split(test_size=0.15, seed=4)
    ds["test"] = ds["test"].filter(lambda example: example["cmd_name"] in ds["train"]["cmd_name"])
    print("length of train set is: ", len(ds["train"]))
    print("length of test set: ", len(ds["test"]))

    if context:
        return add_context(ds["train"]), add_context(ds["test"])
    else:
         return ds["train"], ds["test"]


def add_context(data):
    
    nl_context = []
    for i in range(0,len(data)):
        j,context = 0,""
        while j < len(data[i]["oracle_man"]):
            if data[i]["oracle_man"][j] in context_file:
                new_context = "manual " + str(j) + ": " + context_file[data[i]["oracle_man"][j]]
                if len(tokenizer.encode(context + new_context)) < 2000:
                    context += new_context + "\n"
                else:
                    break
            j += 1
        context += 'query: ' + data[i]["nl"]
        nl_context.append(context)

    data = data.add_column("nl_context", nl_context)
    return(data)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")

def add_special_tokens():
	# special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>', 'bos_token':'<|start|>', 'additional_special_tokens':['<|query|>']}
    special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}

    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer

def preprocess_function(examples):
    model_inputs = tokenizer(text=examples["nl_cmd"], truncation=True, max_length=256)

    labels = tokenizer(text_target=examples["cmd_eos"], truncation=True, max_length=256)
    # model_inputs["labels"] = labels["input_ids"][1:]
    sep_id = model_inputs["input_ids"].index(tokenizer.sep_token_id)
    model_inputs["labels"] = model_inputs["input_ids"][sep_id+1:]

    return model_inputs

def add_sep_token(data):

    input_concat = []
    cmd_complete = []
    for i in range(0,len(data)):
        input_concat.append(data[i]["nl"] + '<|sep|>' + data[i]["cmd"] + tokenizer.eos_token)
        cmd_complete.append(" " + data[i]["cmd"] + tokenizer.eos_token)

    data = data.add_column("nl_cmd", input_concat)
    data = data.add_column("cmd_eos", cmd_complete)

    return data


def padding_masking(element, block_size=256, padding_side="right"):
    inputs = element["input_ids"]
    outputs = element["labels"]
    if padding_side == 'left':
            input_ids = [tokenizer.pad_token_id] * (block_size - len(inputs)) + inputs
            attention_mask = [0] * (block_size - len(inputs)) + [1] * len(inputs)
            labels = [-100] * (block_size - len(outputs)) + outputs[i]
    else:
            input_ids = inputs + [tokenizer.pad_token_id] * (block_size - len(inputs))
            attention_mask = [1] * len(inputs) + [0] * (block_size - len(inputs))
            labels = [-100] * (len(inputs) - len(outputs)) + outputs + [-100] * (block_size - len(inputs))
            
    return {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "labels" : labels
    }


def main():

    random_seed = 2
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    train_set, test_set = prepare_data()
    add_special_tokens()
    tldr_train = add_sep_token(train_set)
    tldr_val = add_sep_token(test_set)
    print(repr(tldr_train[0]["nl_cmd"]))
    print(repr(tldr_train[0]["cmd_eos"]))
    tokenized_dataset = tldr_train.map(preprocess_function)
    tokenized_dataset = tokenized_dataset.remove_columns(['question_id', 'nl', 'cmd', 'oracle_man', 'cmd_name', 'tldr_cmd_name', 'manual_exist', 'matching_info', 'nl_cmd','cmd_eos'])
    # tokenized_dataset_val = tokenized_dataset_val.remove_columns(['question_id', 'nl', 'cmd', 'oracle_man', 'cmd_name', 'tldr_cmd_name', 'manual_exist', 'matching_info', 'nl_cmd','cmd_eos'])
    tokenized_dataset = tokenized_dataset.map(padding_masking)

    tokenized_dataset.set_format("torch")
    args = TrainingArguments(
    output_dir="mistral",
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=2,
    # evaluation_strategy="epoch",
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    # save_total_limit=3,
    num_train_epochs=2,
    weight_decay=0.01,
    learning_rate=4e-5,
    lr_scheduler_type="linear",
    bf16=True,
    # fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=default_data_collator,
        train_dataset=tokenized_dataset,
        # eval_dataset=tldr_val
    )

    trainer.train()
    trainer.save_model("/raid/nlp/sameer/NL2bash/model_checkpoints/starcoder-3b-base")

if __name__ == '__main__':
	main()

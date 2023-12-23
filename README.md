Execution instructions for Colbert: 
Refer to folder https://github.com/sameerp30/Structured-generation/tree/main/Colbert_files


**SBert_top_3.py** - Give the argument of num_negatives in the following format. Python3 <file name> <num_negatives>

**Colbert_train_triple.py** - Give the argument of num_negatives in the following format. Python3 <file name> <num_negatives>

**prepare_index_data.py**

**prepare_train_data.py**

Finally, the training file. 
**train_colbert.py**: Give path of indexing, query file. 

Give the path of the Colbert-v2 checkpoint (https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz) for the argument checkpoint.

**Data Folder**
All the scrapped and alternate flags data is in the folder https://github.com/sameerp30/Structured-generation/tree/main/flags_and_description_data. finals_index.tsv file in the same folder is a corpus of descriptions used for IR

## Installation

Install dependencies

```
pip install -r req.txt
```

Install guidance port for gpt neo
```
cd guidance-port-gpt-neo
pip install -e .
```

## Running the tool
We provide option of running in 3 configurations:

### Base model only

```
python base.py "EleutherAI/gpt-neo-2.7B" "- name: configure aws s3 account on ibm spectrum"
```

### Base model + IR

```
python base_ir.py "EleutherAI/gpt-neo-2.7B" "- name: configure aws s3 account on ibm spectrum" "['ibm.spectrum_virtualize.ibm_sv_manage_awss3_cloudaccount', 'ansible.builtin.file']"
```

### Base model + IR + CD

```
python base_ir_cd.py "EleutherAI/gpt-neo-2.7B" "- name: configure aws s3 account on ibm spectrum" "['ibm.spectrum_virtualize.ibm_sv_manage_awss3_cloudaccount', 'ansible.builtin.file']"
```

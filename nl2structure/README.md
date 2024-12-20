## Installation

Install dependencies

```
pip install -r req.txt
```

Install guidance port for gpt neo
```
cd guidance-port-for-gptneo
pip install -e .
```

## Running the tool
We provide option of running in 3 configurations:

### Base model only

```
python base.py "EleutherAI/gpt-neo-2.7B" "- name: configure aws s3 account on ibm spectrum"
```

output
```
- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum

- name: configure aws s3 account on ibm spectrum
```

### Base model + IR + CD

```
cd guidance_pipeline
python base_ir_cd.py "EleutherAI/gpt-neo-2.7B" "- name: configure aws s3 account on ibm spectrum" "['ibm.spectrum_virtualize.ibm_sv_manage_ip_partnership', 'ibm.spectrum_virtualize.ibm_sv_manage_awss3_cloudaccount']"
```

output
```
- name: configure aws s3 account on ibm spectrum
  ibm.spectrum_virtualize.ibm_sv_manage_awss3_cloudaccount:
    name: "AWS S3 Account"
    state: present
    clustername: "ibm-spectrum"
  register: result
```

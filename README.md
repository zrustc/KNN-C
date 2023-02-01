Official implementation of our paper "Improving Few-Shot Performance of Language Models via Nearest Neighbor Calibration"

## Requirements 

* python==3.6
* torch==1.10.1
* torch-scatter==2.0.7
* transformers==4.18.0

## Datasets

We used the same datasets from Karimi Mahabadi et al. (ACL 2022). 
We first obtain datasets via data pre-processing scripts in https://github.com/facebookresearch/perfect, and then follow cmd to construct datasets (set the right data path before running following scripts).

```
python3 tasks.py
sh data_process.sh
```

## Running

* Download the pre-trained models (e.g., roberta-large)
* run ```sh task_run.sh```

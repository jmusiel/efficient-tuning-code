# NLP 11711 Final Project Repo
Code repo for the final group project for NLP 11711 Fall 2021

All credit for this implementation goes to the principal authors of the original version of this codebase: Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig

This code is really a fork of the code used for [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf). 

Code from the Project 3 [repo](https://github.com/jmusiel/revisit-bert-finetuning) was also used.

Original README text:
# Towards a Unified Framework of Parameter-Efficient Transfer Learning
This code is fork of [huggingface transformer](https://huggingface.co/transformers/). 


### Usage

Run MAM Adapter on XSum:

```bash
bash exp_xsum/run_bart_trainer.sh
```



Run MAM Adapter on en-ro:

```bash
bash exp_mt/run_mbart_trainer.sh
```



Run MAM Adapter on MNLI or SST2:

```bash
bash exp_glue/run_glue_trainer.sh
```


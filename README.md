# NLP 11711 Final Project Repo
Code repo for the final group project for NLP 11711 Fall 2021

All credit for this implementation goes to the principal authors of the original version of this codebase: 

Junxian He*
Chunting Zhou*
Xuezhe Ma
Taylor Berg-Kirkpatrick
Graham Neubig. 

*: Equal Contribution

Code used with permission from Junxian He and Chunting Zhou.

The majority of this codebase is a fork of the code used for [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf), which in turn is a fork of [huggingface transformer](https://huggingface.co/transformers/).

Besides some tweaks and bugfixes to the original codebase, all the code written for this NLP 11711 final project is contained within the [low_resource_glue_data directory](https://github.com/jmusiel/efficient-tuning-code/tree/main/low_resource_glue_data).

Additional code written for this final project, which was used for the BERT low resource finetuning comparison experiments, can be found in the Project 3 repo in the [low_resource_glue branch](https://github.com/jmusiel/revisit-bert-finetuning/tree/low_resource_glue), specifically contained within the [low_resource directory](https://github.com/jmusiel/revisit-bert-finetuning/tree/low_resource_glue/low_resource).

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


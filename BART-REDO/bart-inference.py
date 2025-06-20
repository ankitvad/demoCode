import torch
import numpy as np
import nltk
import pandas as pd
import pickle as p
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, TrainingArguments, pipeline, logging, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from evaluate import load
import sys
from tqdm import tqdm
import os

# TOKENIZERS_PARALLELISM=False

# TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1 python bart-inference.py ft-corrup-NI/

# TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python bart-inference.py ft-corrup-I/

# Processed = train_proc = NI, #Original = train.csv

op_dir = sys.argv[1]

deviceC = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "/datasets/ankitUW/trained_model/bart-ft/FT/" + sys.argv[1] + "checkpoint-135/"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


"""## Model"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.generation_config.max_new_tokens = 100
#model.generation_config.min_new_tokens = 5

op_dir = "/datasets/ankitUW/trained_model/bart-ft/FT/" + op_dir

pipe = pipeline(task="text2text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 100, batch_size=32, device = deviceC)

dev = "/datasets/ankitUW/resources/grndtr_data/EN/dev.csv"
dev_sents = open(dev,"r").read().strip().split("\n")
s_t_h = {"s":[], "t":[], "h":[]}

for d in tqdm(dev_sents):
	tmp = d.split("|\t|")
	assert(len(tmp) == 2)
	s = tmp[0]
	t = tmp[1]
	result = pipe(s)
	h = result[0]['generated_text']
	s_t_h["s"].append(s)
	s_t_h["t"].append(t)
	s_t_h["h"].append(h)

p.dump(s_t_h, open(op_dir+"s_t_h_infer_200.p", "wb"))

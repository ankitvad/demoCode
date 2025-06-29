import torch
import numpy as np
import pandas as pd
import pickle as p
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, TrainingArguments, pipeline, logging, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from evaluate import load
import sys
from tqdm import tqdm
import os, gc
import glob
import time

# TOKENIZERS_PARALLELISM=False

# TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python bart-inference-all.py models-LT/ ip_file.txt

model_name = sys.argv[1]
all_locations = glob.glob(model_name + "*/checkpoint-*/")

ip_file = sys.argv[2]
ip_file = open(ip_file, "r").read().strip().split("\n")

#op_file = sys.argv[3]

deviceC = 'cuda' if torch.cuda.is_available() else 'cpu'


for m in tqdm(all_locations):
	print("Loading model from: ", m)
	model_name = m
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "right"
	"""## Model"""
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
	model.generation_config.max_new_tokens = 200
	model.generation_config.num_beams = 6
	#model.generation_config.min_new_tokens = 5
	pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, batch_size=20, device=deviceC)
	gen_answer = [out["generated_text"] for out in pipe(ip_file)]
	# Write out in text file.
	op_dir = model_name + "gen_output_beam6.txt"
	writeOut = open(op_dir, "w")
	for g in gen_answer:
		tmp = writeOut.write(g + "\n")
	writeOut.close()
	#Wait for 30 seconds
	time.sleep(30)
	print("Done writing out the generated answers to the file: ", op_dir)
	del model, tokenizer, pipe, gen_answer
	gc.collect()
	torch.cuda.empty_cache()


'''

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
'''
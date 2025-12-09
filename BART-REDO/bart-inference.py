import torch
import numpy as np
import pandas as pd
import pickle as p
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from datasets import Dataset
from evaluate import load
import sys
import glob
#from tqdm import tqdm
import os
import gc

model_loc = sys.argv[1]
ip_file = sys.argv[2]
op_file = "gen_output_beam5.txt"

#Model Location has many checkpoints. Use glob to load all checpoints and then choose the oldest and latest one out of all of them.

all_checkpoints = glob.glob(model_loc.strip("/") + "/checkpoint-*/")
start_val = -1
model_name = ""

for c in all_checkpoints:
	val = int(c.strip("/").split("-")[-1])
	if val > start_val:
		start_val = val
		model_name = c


deviceC = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


"""## Model"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.generation_config.max_new_tokens = 200
model.generation_config.num_beams = 5
model.config.num_beams = 5


op_dir = model_name + op_file

pipe = pipeline(task="text2text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 200, batch_size=20, device = deviceC)

ip_file = open(ip_file, "r").read().strip().split("\n")
ip_dataset = [{"text": text} for text in ip_file]
del ip_file
gc.collect()

gen_answer = []

for out in tqdm(pipe(KeyDataset(ip_dataset, "text")), total=len(ip_dataset)):
	gen_answer.append(out[0]["generated_text"])
	#print(out)
	#break

#Write out in text file.
writeOut = open(op_dir, "w")
for g in gen_answer:
	tmp = writeOut.write(g + "\n")
writeOut.close()

print("Done writing out the generated answers to the file: ", op_dir)

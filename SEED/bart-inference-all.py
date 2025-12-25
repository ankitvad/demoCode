import torch
import numpy as np
import pandas as pd
import pickle as p
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, TrainingArguments, pipeline, logging, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from evaluate import load
import sys
from tqdm import tqdm
from jiwer import wer
import os, gc
import glob
import time
from utils import *

# TOKENIZERS_PARALLELISM=False
# TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python bart-inference-all.py models-LT/ ip_file.txt 0-2

model_name = sys.argv[1]

ip_file = sys.argv[2]
ip_file = open(ip_file, "r").read().strip().split("\n")
trg_file = sys.argv[3]
trg_file = open(trg_file, "r").read().strip().split("\n")

indx_range = sys.argv[4].strip().split("-")
assert len(indx_range) == 2, "Please provide a valid range for the model checkpoints."
indx_range = [int(i) for i in indx_range]

all_locations = glob.glob(model_name.rstrip("/") + "/*/*/checkpoint-*/")
all_locations = sorted(all_locations)
#all_locations = all_locations[indx_range[0]:indx_range[1]]
#op_file = sys.argv[3]
deviceC = 'cuda' if torch.cuda.is_available() else 'cpu'

#Check if RUNNING_DONE file exists for all locations and remove those folders from all_locations.

all_locations_update = []
for m in all_locations:
	if os.path.exists(m + "RUNNING_DONE"):
		print("The RUNNING_DONE file exists for the model: ", m)
		print("Skipping this model as it has already been processed.")
	else:
		all_locations_update.append(m)

all_locations_update = all_locations_update[indx_range[0]:indx_range[1]]

all_locations = all_locations_update

#Check that RUNNING_NOW file does not exist and save it for al locations in all_locations.
for m in all_locations:
	if os.path.exists(m + "RUNNING_NOW"):
		print("The RUNNING_NOW file exists for the model: ", m)
		print("Please delete the RUNNING_NOW file to run the inference again.")
		sys.exit(0)
	running_now_file = open(m + "RUNNING_NOW", "w")
	running_now_file.close()


def generate_metrics(man, aws, hyp, writeOut):
	writeOut = open(writeOut, "w")
	p_correct, p_ignore, p_introd, wer_am, wer_hm = getEvalScore(man, aws, hyp)
	writeOut.write(f"p_correct: {p_correct}, p_ignore: {p_ignore}, p_introd: {p_introd}, wer_am: {wer_am}, wer_hm: {wer_hm}\n\n")
	#Remove Insertion Edits and Recalculate:
	writeOut.write("After removing insertion edits:\n\n")
	hyp_NI = [remove_insert_edits(hyp[i], aws[i], do_test=False) for i in range(len(hyp))]
	p_correct, p_ignore, p_introd, wer_am, wer_hm = getEvalScore(man, aws, hyp_NI)
	writeOut.write(f"p_correct: {p_correct}, p_ignore: {p_ignore}, p_introd: {p_introd}, wer_am: {wer_am}, wer_hm: {wer_hm}\n\n")
	writeOut.close()
	return True

for m in tqdm(all_locations):
	assert (os.path.exists(m+"RUNNING_DONE") == False)
	print("Loading model from: ", m)
	model_name = m
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "right"
	#Save the RUNNING_NOW file to indicate that the model is currently being processed.
	#running_now_file = open(model_name + "RUNNING_NOW", "w")
	#running_now_file.close()
	"""## Model"""
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
	model.generation_config.max_new_tokens = 200
	model.generation_config.num_beams = 5
	#model.generation_config.min_new_tokens = 5
	pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, batch_size=32, device=deviceC)
	gen_answer = []
	ip_dataset = [{"text": text} for text in ip_file]
	for out in tqdm(pipe(KeyDataset(ip_dataset, "text")), total=len(ip_dataset)):
		gen_answer.append(out[0]["generated_text"])
	# Write out in text file.
	op_dir = model_name.split("checkpoint-")[0] + "gen_output_beam5.txt"
	writeOut = open(op_dir, "w")
	for g in gen_answer:
		tmp = writeOut.write(g + "\n")
	writeOut.close()
	#Wait for 30 seconds
	time.sleep(30)
	print("Done writing out the generated answers to the file: ", op_dir)
	#Generate the evaluation scores:
	op_loc = model_name.split("checkpoint-")[0] + "wer_scores.txt"
	tmp = generate_metrics(man = trg_file, aws = ip_file, hyp = gen_answer, writeOut = op_loc)
	print("Done writing out the evaluation scores to the file: ", op_loc)
	del model, tokenizer, pipe, gen_answer
	gc.collect()
	torch.cuda.empty_cache()
	#Delete RUNNING_NOW file to indicate that the model has been processed, and save RUNNING_DONE file to indicate that the model has been processed.
	os.remove(m + "RUNNING_NOW")
	running_done_file = open(m + "RUNNING_DONE", "w")
	running_done_file.close()
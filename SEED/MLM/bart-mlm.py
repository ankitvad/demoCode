import torch
import numpy as np
import pandas as pd
import pickle as p
from transformers import (
	AutoModelForSeq2SeqLM, 
	DataCollatorForSeq2Seq, 
	AutoTokenizer, 
	Seq2SeqTrainer, 
	Seq2SeqTrainingArguments, 
	EarlyStoppingCallback, 
	set_seed
)
from datasets import Dataset
from evaluate import load
import sys
import glob
import random
from tqdm import tqdm
import os
import gc
import argparse
# Note: Ensure utils.py exists in your path if you have specific scoring functions there
# from utils import * device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Arguments ---
parser = argparse.ArgumentParser(description="BART Self-Supervised Text-Infilling Script")
parser.add_argument('--train_src', type=str, help='Path to the raw text file for training')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--dev_src', type=str, help='Path to the raw text file for development')
parser.add_argument('--op_dir', type=str, help='Output directory')
parser.add_argument('--resume_train', type=int, choices=[0, 1], default=0)
parser.add_argument('--model_path', type=str, default='facebook/bart-base', help='BART model to initialize from')
parser.add_argument('--early_stop', action='store_true')
parser.add_argument('--mlm_prob', type=float, default=0.15, help='Probability of masking tokens')
parser.add_argument('--poisson_lambda', type=float, default=3.0, help='Lambda for Poisson span length')

args = parser.parse_args()

# --- Seed Initialization ---
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
set_seed(args.seed)

op_dir = args.op_dir.rstrip("/") + "/" + str(args.seed) + "/"

resumeTrain = True if args.resume_train == 1 else False
chk_chkp = glob.glob(op_dir + "checkpoint*")
if chk_chkp:
	resumeTrain = True

# --- Data Loading ---
def load_raw_data(path):
	"""Loads raw lines as both source and target (target is the ground truth)."""
	lines = open(path, "r", encoding="utf-8").read().strip().split("\n")
	df = pd.DataFrame({"text": lines})
	return df

train_df = load_raw_data(args.train_src)
val_df = load_raw_data(args.dev_src)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df).shuffle(seed=args.seed).select(range(min(1000, len(val_df))))

# --- Tokenizer & Model Setup ---
model_name = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# --- Text Infilling Logic ---
def apply_text_infilling(tokens, tokenizer, mlm_probability=0.15, poisson_lambda=3.0):
	"""Corrupts input tokens by replacing spans with a single mask token."""
	input_ids = tokens.copy()
	sz = len(input_ids)
	num_mask = int(sz * mlm_probability)
	
	# Identify non-special tokens
	special_ids = set(tokenizer.all_special_ids)
	indices = [i for i, idx in enumerate(input_ids) if idx not in special_ids]
	
	masked_indices = set()
	num_masked_tokens = 0
	
	while num_masked_tokens < num_mask and indices:
		span_len = np.random.poisson(poisson_lambda)
		if span_len == 0:
			continue
		
		start_idx = np.random.choice(indices)
		end_idx = min(start_idx + span_len, sz)
		
		for i in range(start_idx, end_idx):
			masked_indices.add(i)
			if i in indices: indices.remove(i)
		
		num_masked_tokens += span_len
	# Build the noisy sequence: replace contiguous masked spans with a SINGLE mask token
	noisy_ids = []
	in_mask_span = False
	
	for i in range(sz):
		if i in masked_indices:
			if not in_mask_span:
				noisy_ids.append(tokenizer.mask_token_id)
				in_mask_span = True
		else:
			noisy_ids.append(input_ids[i])
			in_mask_span = False
			
	return noisy_ids

def preprocess_function(examples):
	# 1. Target is the clean text
	labels = tokenizer(examples['text'], max_length=150, truncation=True)
	
	# 2. Source is the corrupted version of the clean text
	noisy_input_ids = []
	for ids in labels['input_ids']:
		noisy_input_ids.append(apply_text_infilling(ids, tokenizer, args.mlm_prob, args.poisson_lambda))
	
	# Pad manually or handle via DataCollator
	model_inputs = {"input_ids": noisy_input_ids, "labels": labels['input_ids']}
	return model_inputs

# --- Processing ---
train_ds_tok = train_dataset.map(preprocess_function, batched=True, num_proc=8)
val_ds_tok = val_dataset.map(preprocess_function, batched=True, num_proc=8)

del train_df, val_df, train_dataset, val_dataset
gc.collect()

# --- Metrics ---
metric_wer = load('wer')

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	if isinstance(preds, tuple):
		preds = preds[0]
	
	preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	
	result = metric_wer.compute(predictions=decoded_preds, references=decoded_labels)
	return {"wer": result}

# --- Model Initialization ---
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.config.max_length = 150
model.config.num_beams = 4

assert(model.config.encoder_layers in [6, 12])
if model.config.encoder_layers == 12:
	bart_model_type = "bart-large"
	LRVal = 1e-5
	train_batch_size = 16
	grad_accum_steps = 8
	eval_batch_size = 20
	if "barthez" in model_name.lower():
		train_batch_size = 8
		grad_accum_steps = 16
		eval_batch_size = 10
else:
	bart_model_type = "bart-base"
	LRVal = 3e-5
	train_batch_size = 32
	grad_accum_steps = 4
	eval_batch_size = 20

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# --- Training Config ---
training_args = Seq2SeqTrainingArguments(
	output_dir=op_dir,
	learning_rate=LRVal,
	per_device_train_batch_size=train_batch_size,
	per_device_eval_batch_size=eval_batch_size,
	gradient_accumulation_steps=grad_accum_steps,
	eval_strategy="steps",
	save_strategy="steps",
	save_steps=3000,
	eval_steps=3000,
	save_total_limit=2,
	weight_decay=0.01,
	load_best_model_at_end=True,
	num_train_epochs=5,
	fp16=torch.cuda.is_available(),
	predict_with_generate=True,
	logging_steps=100,
	warmup_steps=500,
	report_to="none"
)

callbacks = [EarlyStoppingCallback(3, 0.0001)] if args.early_stop else []

trainer = Seq2SeqTrainer(
	model=model,
	args=training_args,
	train_dataset=train_ds_tok,
	eval_dataset=val_ds_tok,
	tokenizer=tokenizer,
	data_collator=data_collator,
	compute_metrics=compute_metrics,
	callbacks=callbacks
)

# --- Execute ---
print(f"Starting Training. Resume: {resumeTrain}")
trainer.train(resume_from_checkpoint=resumeTrain)
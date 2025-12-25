import torch
import numpy as np
import pandas as pd
import pickle as p
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed
from datasets import Dataset
from evaluate import load
import sys
import glob
import random
from tqdm import tqdm
import os
import gc
import argparse
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description="BART Fine-tuning and Inference Script")
parser.add_argument('--train_src', type=str, help='Path to the training source file')
parser.add_argument('--train_trg', type=str, help='Path to the training target file')
parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
parser.add_argument('--dev_src', type=str, help='Path to the development source file')
parser.add_argument('--dev_trg', type=str, help='Path to the development target file')
parser.add_argument('--op_dir', type=str, help='Output directory for the model')
parser.add_argument('--start_model', type=str, help='Starting weights for the model or path to local model.')
parser.add_argument('--early_stop', action='store_true', help='Whether to use early stopping during training')
args = parser.parse_args()

# Set Seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
set_seed(args.seed)

# Define Output Directory with Seed
op_dir = args.op_dir.rstrip("/") + "/" + str(args.seed) + "/"
if not os.path.exists(op_dir):
	os.makedirs(op_dir)

#Model Location has many checkpoints. Use glob to load all checpoints and then choose the oldest and latest one out of all of them.

all_checkpoints = glob.glob(args.start_model.rstrip("/") + "/checkpoint-*/")
start_val = -1
model_name = ""

for c in all_checkpoints:
	val = int(c.rstrip("/").split("-")[-1])
	if val > start_val:
		start_val = val
		model_name = c

def loadData(s, t):
	df = pd.DataFrame()
	src = open(s, "r").read().strip().split("\n")
	trg = open(t, "r").read().strip().split("\n")
	assert len(src) == len(trg), "Source and target files must have the same number of lines."
	df["src"] = src
	df["trg"] = trg
	return df

def generate_metrics(man, aws, hyp, writeOut_path):
	with open(writeOut_path, "w") as f:
		p_correct, p_ignore, p_introd, wer_am, wer_hm = getEvalScore(man, aws, hyp)
		f.write(f"p_correct: {p_correct}, p_ignore: {p_ignore}, p_introd: {p_introd}, wer_am: {wer_am}, wer_hm: {wer_hm}\n\n")
		f.write("After removing insertion edits:\n\n")
		hyp_NI = [remove_insert_edits(hyp[i], aws[i], do_test=False) for i in range(len(hyp))]
		p_correct, p_ignore, p_introd, wer_am, wer_hm = getEvalScore(man, aws, hyp_NI)
		f.write(f"p_correct: {p_correct}, p_ignore: {p_ignore}, p_introd: {p_introd}, wer_am: {wer_am}, wer_hm: {wer_hm}\n\n")
	return True

# Load Datasets
train_ds_raw = loadData(args.train_src, args.train_trg)
val_ds_raw = loadData(args.dev_src, args.dev_trg)

train_dataset = Dataset.from_pandas(train_ds_raw)
val_dataset_subset = Dataset.from_pandas(val_ds_raw).shuffle(seed=args.seed).select(range(min(1000, len(val_ds_raw))))


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

def preprocess_function(examples):
	model_inputs = tokenizer(examples['src'], max_length=150, truncation=True)
	labels = tokenizer(text_target=examples['trg'], max_length=150, truncation=True)
	model_inputs['labels'] = labels['input_ids']
	return model_inputs

train_ds_tok = train_dataset.map(preprocess_function, batched=True, num_proc=20)
val_ds_tok = val_dataset_subset.map(preprocess_function, batched=True, num_proc=20)

metric_wer = load('wer')

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	if isinstance(preds, tuple): preds = preds[0]
	preds[preds == -100] = tokenizer.pad_token_id
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	result = metric_wer.compute(predictions=decoded_preds, references=decoded_labels)
	return {"wer_loss": result}

# Initialize Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.num_beams = 4

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
	output_dir=op_dir,
	learning_rate=3e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	gradient_accumulation_steps=4,
	eval_strategy="steps",
	save_strategy="steps",
	save_steps=100,
	eval_steps=100,
	save_total_limit=2,
	load_best_model_at_end=True,
	metric_for_best_model="wer_loss",
	greater_is_better=False,
	num_train_epochs=10,
	fp16=torch.cuda.is_available(),
	predict_with_generate=True,
	group_by_length=True,
	report_to="none",
	save_only_model=True
)

callbacks = [EarlyStoppingCallback(5, 0.001)] if args.early_stop else []

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

# Training Phase
trainer.train()

# --- Inference Phase ---
print("Starting Inference on full validation set...")
full_val_dataset = Dataset.from_pandas(val_ds_raw)
full_val_tok = full_val_dataset.map(preprocess_function, batched=True, num_proc=20)

trainer.args.per_device_eval_batch_size = 32
predictions = trainer.predict(full_val_tok)
preds = predictions.predictions
labels = predictions.label_ids

if isinstance(preds, tuple): preds = preds[0]
preds[preds == -100] = tokenizer.pad_token_id
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

# Save predictions
gen_file = os.path.join(op_dir, "gen_output_beam4.txt")
with open(gen_file, "w") as f:
	for line in decoded_preds:
		f.write(line.strip() + "\n")

# Save detailed metrics
metrics_file = os.path.join(op_dir, "wer_scores.txt")
generate_metrics(man=val_ds_raw["trg"].tolist(),
				aws=val_ds_raw["src"].tolist(),
				hyp=decoded_preds,
				writeOut_path=metrics_file)

print(f"Workflow complete. Results saved in {op_dir}")
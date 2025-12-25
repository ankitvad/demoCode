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
#LIbrary to load better CLI parameter arguments
import argparse
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Take in the arguments for - training source, target and dev source and target and the location to save the model output and a boolean value whether to resume training or start from scratch.

parser = argparse.ArgumentParser(description="BART pre-training Script")
parser.add_argument('--train_src', type=str, help='Path to the training source file')
parser.add_argument('--seed', type=int, help='Random seed for initialization')
parser.add_argument('--train_trg', type=str, help='Path to the training target file')
parser.add_argument('--dev_src', type=str, help='Path to the development source file')
parser.add_argument('--dev_trg', type=str, help='Path to the development target file')
parser.add_argument('--op_dir', type=str, help='Output directory for the model')
parser.add_argument('--resume_train', type=int, choices=[0, 1], help='Resume training (1) or start fresh (0)')
parser.add_argument('--model_path', type=str, default='', help='Path to a pre-trained model (optional)')
parser.add_argument('--early_stop', action='store_true', help='Whether to use early stopping during training')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
set_seed(args.seed)

op_dir = args.op_dir.strip("/") + str("/") + str(args.seed) + "/"

resumeTrain = args.resume_train
if resumeTrain == 1:
	resumeTrain = True
else:
	resumeTrain = False

chk_chkp = glob.glob(op_dir+"checkpoint*")
if chk_chkp:
	resumeTrain = True
else:
	resumeTrain = False


def loadData(s,t):
	df = pd.DataFrame()
	src = open(s,"r").read().strip().split("\n")# s
	trg = open(t,"r").read().strip().split("\n")# t
	assert len(src) == len(trg), "Source and target files must have the same number of lines."
	df["src"] = src
	df["trg"] = trg
	return df


train_ds = loadData(args.train_src, args.train_trg)
val_ds = loadData(args.dev_src, args.dev_trg)

train_dataset = Dataset.from_pandas(train_ds)
val_dataset = Dataset.from_pandas(val_ds).shuffle(seed=args.seed).select(range(1000))

model_name = args.model_path#"/home/avadehra/scribendi/dwnldModel/bart-large/"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

def preprocess_function(examples):
	model_inputs = tokenizer(examples['src'], max_length=150, truncation=True)
	labels = tokenizer(text_target=examples['trg'], max_length=150, truncation=True)
	model_inputs['labels'] = labels['input_ids']
	return model_inputs

train_ds_tok = train_dataset.map(preprocess_function, batched=True, num_proc=60)#os.cpu_count() - 5)
val_ds_tok = val_dataset.map(preprocess_function, batched=True, num_proc=60)#os.cpu_count() - 5)

del train_ds, val_ds, train_dataset, val_dataset
gc.collect()

metric_wer = load('wer')

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	# decode preds and labels
	preds[preds == -100] = tokenizer.pad_token_id
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	# rougeLSum expects newline after each sentence
	decoded_preds = [pred.strip() for pred in decoded_preds]
	decoded_labels = [label.strip() for label in decoded_labels]
	result = metric_wer.compute(predictions=decoded_preds, references=decoded_labels)
	return {"wer_loss" : result}

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

"""## Model"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.generation_config.max_new_tokens = 150
model.generation_config.min_new_tokens = 2
model.generation_config.early_stopping = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.num_beams = 4
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.num_beams = 4

# Batching function
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

#early_stop = EarlyStoppingCallback(5, 0.0001)
early_stop = EarlyStoppingCallback(5, 0.001)

# Define arguments of the finetuning
training_args = Seq2SeqTrainingArguments(
	output_dir=op_dir,
	learning_rate=1e-7,
	per_device_train_batch_size=8,# batch size for train
	per_device_eval_batch_size=16,
	gradient_accumulation_steps=4,
	eval_strategy="steps",
	save_strategy="steps",
	save_steps = 100,
	eval_steps = 100,
	save_total_limit=1,# num of checkpoints to save
	#metric_for_best_model="wer_loss",
	load_best_model_at_end = True,
	num_train_epochs=30,
	fp16=False,
	predict_with_generate=True,
	dataloader_num_workers=16,
	greater_is_better=False,
	group_by_length=True,
	report_to="none",
	save_only_model=True
)

assert(args.early_stop in [True, False]), "early_stop argument must be a boolean."

if args.early_stop:
	print("Using Early Stopping Callback")
	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_ds_tok,
		eval_dataset=val_ds_tok,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		callbacks=[early_stop])
else:
	print("Not Using Early Stopping Callback")
	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_ds_tok,
		eval_dataset=val_ds_tok,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics)


trainer.train(resume_from_checkpoint=resumeTrain)

#Perform Inference on the validation set and save the output.

val_ds = loadData(args.dev_src, args.dev_trg)
val_dataset = Dataset.from_pandas(val_ds)
val_ds_tok = val_dataset.map(preprocess_function, batched=True, num_proc=60)#os.cpu_count() - 5)
#Update the batch size for inference
trainer.args.per_device_eval_batch_size = 32
# Get predictions and labels
predictions = trainer.predict(val_ds_tok)
preds = predictions.predictions
labels = predictions.label_ids
# Decode texts - handle None values properly
preds[preds == -100] = tokenizer.pad_token_id
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
# Filter out any None values that might still exist
decoded_preds = [pred.strip() for pred in decoded_preds]
decoded_labels = [label.strip() for label in decoded_labels]
# Write out the predictions to a file
op_dir2 = op_dir.rstrip("/") + "/gen_output_beam4.txt"
writeOut = open(op_dir2, "w")
for g in decoded_preds:
	tmp = writeOut.write(g + "\n")
writeOut.close()
op_dir2 = op_dir.rstrip("/") + "/wer_scores.txt"
tmp = generate_metrics(man = val_ds["trg"].tolist(), aws = val_ds["src"].tolist(), hyp = decoded_preds, writeOut = op_dir2)
print("Done writing out the evaluation scores to the file: ", op_dir2)



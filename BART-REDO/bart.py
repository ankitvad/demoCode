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
#LIbrary to load better CLI parameter arguments
import argparse

#TOKENIZERS_PARALLELISM=False


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Take in the arguments for - training source, target and dev source and target and the location to save the model output and a boolean value whether to resume training or start from scratch.

parser = argparse.ArgumentParser(description="BART pre-training Script")
parser.add_argument('--train_src', type=str, help='Path to the training source file')
parser.add_argument('--train_trg', type=str, help='Path to the training target file')
parser.add_argument('--dev_src', type=str, help='Path to the development source file')
parser.add_argument('--dev_trg', type=str, help='Path to the development target file')
parser.add_argument('--op_dir', type=str, help='Output directory for the model')
parser.add_argument('--resume_train', type=int, choices=[0, 1], help='Resume training (1) or start fresh (0)')
args = parser.parse_args()
#Sample command to run the script:
#TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1 python bart.py --train_src /datasets/ankitUW/resources/grndtr_data/EN/train_proc.csv --train_trg /datasets/ankitUW/resources/grndtr_data/EN/train_proc.csv --dev_src /datasets/ankitUW/resources/grndtr_data/EN/dev.csv --dev_trg /datasets/ankitUW/resources/grndtr_data/EN/dev.csv --op_dir ft-corrup-NI/ --resume_train 0


resumeTrain = args.resume_train
if resumeTrain == 1:
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
val_dataset = Dataset.from_pandas(val_ds)

model_name = "/datasets/model/bart-large/"

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

metric_wer = load('wer')

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	# decode preds and labels
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	# rougeLSum expects newline after each sentence
	decoded_preds = [pred.strip() for pred in decoded_preds]
	decoded_labels = [label.strip() for label in decoded_labels]
	result = metric_wer.compute(predictions=decoded_preds, references=decoded_labels)
	return {"wer_loss" : result}


"""## Model"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.generation_config.max_new_tokens = 150
model.generation_config.min_new_tokens = 2

# Batching function
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

op_dir = "/datasets/ankitUW/redoBART/" + op_dir

# Define arguments of the finetuning
training_args = Seq2SeqTrainingArguments(
	output_dir=op_dir,
	#evaluation_strategy='epoch',
	learning_rate=3e-5,
	per_device_train_batch_size=16,# batch size for train
	per_device_eval_batch_size=16,
	gradient_accumulation_steps=8,
	eval_strategy="steps",
	save_strategy="steps",
	save_steps = 10000,
	eval_steps = 10000,
	save_total_limit=3,# num of checkpoints to save
	load_best_model_at_end = True,
	num_train_epochs=6,
	fp16=False,
	predict_with_generate=True,
	dataloader_num_workers=16,
	greater_is_better=False,
	group_by_length=True,
	report_to="none"
)

trainer = Seq2SeqTrainer(
	model=model,
	args=training_args,
	train_dataset=train_ds_tok,
	eval_dataset=val_ds_tok,
	tokenizer=tokenizer,
	data_collator=data_collator,
	compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=resumeTrain)

'''
pipe = pipeline(task="text2text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 72, batch_size=32)

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

p.dump(s_t_h, open(op_dir+"s_t_h_infer.p", "wb"))
'''
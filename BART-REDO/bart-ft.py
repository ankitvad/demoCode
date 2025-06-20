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

# TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1 python bart-ft.py /datasets/ankitUW/resources/grndtr_data/EN/train_proc.csv /datasets/ankitUW/resources/grndtr_data/EN/dev.csv ft-corrup-NI/ 0

# TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python bart-ft.py /datasets/ankitUW/resources/grndtr_data/EN/train.csv /datasets/ankitUW/resources/grndtr_data/EN/dev.csv ft-corrup-I/ 0

#Processed = train_proc = NI, #Original = train.csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = sys.argv[1]#training file
dev = sys.argv[2]#dev file
op_dir = sys.argv[3]

resumeTrain = int(sys.argv[4])
if resumeTrain == 1:
	resumeTrain = True
else:
	resumeTrain = False


def loadData(f):
	df = pd.DataFrame()
	x = open(f,"r").read().strip().split("\n")# s, t
	src = []
	trg = []
	for j in x:
		tmp = j.split("|\t|")
		assert(len(tmp) == 2)
		src.append(tmp[0])
		trg.append(tmp[1])
	'''
	tmpM = 1000
	df["src"] = src[:tmpM]
	df["trg"] = trg[:tmpM]
	'''
	df["src"] = src
	df["trg"] = trg
	return df


train_ds = loadData(train)
val_ds = loadData(dev)
train_dataset = Dataset.from_pandas(train_ds)
val_dataset = Dataset.from_pandas(val_ds)

dataset_type = op_dir.strip("/").split("-")[2]
if dataset_type == "NI":
	model_name = "/datasets/ankitUW/trained_model/bart-ft/syn-corrup-NI/checkpoint-242620/"
elif dataset_type == "I":
	model_name = "/datasets/ankitUW/trained_model/bart-ft/syn-corrup-I/checkpoint-242620/"
else:
	print("ERROR IN DATASET TYPE!")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

def preprocess_function(examples):
	model_inputs = tokenizer(examples['src'], max_length=72, truncation=True)
	labels = tokenizer(text_target=examples['trg'], max_length=72, truncation=True)
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
model.generation_config.max_new_tokens = 72
model.generation_config.min_new_tokens = 5

# Batching function
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

op_dir = "/datasets/ankitUW/trained_model/bart-ft/FT/" + op_dir

# Define arguments of the finetuning
training_args = Seq2SeqTrainingArguments(
	output_dir=op_dir,
	#evaluation_strategy='epoch',
	learning_rate=3e-5,
	per_device_train_batch_size=32,# batch size for train
	per_device_eval_batch_size=32,
	gradient_accumulation_steps=4,
	eval_strategy="epoch",
	save_strategy="epoch",
	#save_steps = 10000,
	#eval_steps = 10000,
	save_total_limit=2,# num of checkpoints to save
	load_best_model_at_end = True,
	num_train_epochs=5,
	fp16=False,
	predict_with_generate=True,
	dataloader_num_workers=32,
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


pipe = pipeline(task="text2text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 72, batch_size=32)

dev = dev.split("/")
dev[-1] = "test.csv"
dev = "/".join(dev)
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

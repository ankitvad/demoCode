import torch
import numpy as np
import pandas as pd
import pickle as p
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed, EncoderDecoderModel
from datasets import Dataset
from evaluate import load
import sys
import glob
from tqdm import tqdm
import os
import gc
#LIbrary to load better CLI parameter arguments
import argparse


parser = argparse.ArgumentParser(description="BERT pre-training Script")
parser.add_argument('--train_src', type=str, help='Path to the training source file')
parser.add_argument('--train_trg', type=str, help='Path to the training target file')
parser.add_argument('--dev_src', type=str, help='Path to the development source file')
parser.add_argument('--dev_trg', type=str, help='Path to the development target file')
parser.add_argument('--op_dir', type=str, help='Output directory for the model')
parser.add_argument('--resume_train', type=int, choices=[0, 1], help='Resume training (1) or start fresh (0)')
parser.add_argument('--model_path', type=str, default='', help='Path to a pre-trained model (optional)')
parser.add_argument('--early_stop', action='store_true', help='Whether to use early stopping during training')
parser.add_argument('--tie_encoder_decoder', action='store_true', help='Whether to tie encoder and decoder weights')

args = parser.parse_args()
op_dir = args.op_dir

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
val_dataset = Dataset.from_pandas(val_ds).shuffle(seed=42).select(range(1000))

model_name = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

def preprocess_function(examples):
	model_inputs = tokenizer(examples['src'], max_length=150, truncation= True, padding="max_length")
	labels = tokenizer(text_target=examples['trg'], max_length=150, truncation=True, padding="max_length")
	#model_inputs["decoder_input_ids"] = labels['input_ids']
	#model_inputs["decoder_attention_mask"] = labels["attention_mask"]
	model_inputs['labels'] = labels['input_ids']#.copy()
	#model_inputs['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in l] for l in model_inputs['labels']]
	return model_inputs

batch_size_val = 8

column_names = train_dataset.column_names
train_ds_tok = train_dataset.map(preprocess_function, batched=True, num_proc=32, remove_columns=column_names)#, batch_size = batch_size_val)
val_ds_tok = val_dataset.map(preprocess_function, batched=True, num_proc=32, remove_columns=column_names)#, batch_size = batch_size_val)
#train_ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "decoder_attention_mask"])
#val_ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "decoder_attention_mask"])

del train_ds, val_ds, train_dataset, val_dataset
gc.collect()
metric_wer = load('wer')

assert(args.tie_encoder_decoder in [True, False])
if args.tie_encoder_decoder:
	print("Tying Encoder and Decoder Weights")
else:
	print("Not Tying Encoder and Decoder Weights")

model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, tie_encoder_decoder=args.tie_encoder_decoder)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# text generation parameters
model.config.max_length = 150
model.config.min_length = 2
model.config.early_stopping = True
model.config.num_beams = 4
model.config.no_repeat_ngram_size = 3

# Define compute metrics function
def compute_metrics(pred):
	labels_ids = pred.label_ids
	pred_ids = pred.predictions
	# decoding predictions and labels
	candidates = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
	labels_ids[labels_ids == -100] = tokenizer.pad_token_id
	references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
	decoded_preds = [pred.strip() for pred in candidates]
	decoded_labels = [label.strip() for label in references]
	result = metric_wer.compute(predictions=decoded_preds, references=decoded_labels)
	return {"wer_loss" : result}

data_collator = DataCollatorForSeq2Seq(
	tokenizer=tokenizer,
	model=model,
	label_pad_token_id=-100,
	#pad_to_multiple_of=8)
	)

training_args = Seq2SeqTrainingArguments(
	output_dir=op_dir,
	remove_unused_columns=True,
	eval_strategy="steps",
	save_strategy="steps",
	per_device_train_batch_size=batch_size_val,
	per_device_eval_batch_size=batch_size_val,
	gradient_accumulation_steps=4,
	learning_rate=1e-7,
	save_steps = 100,
	eval_steps = 100,
	save_total_limit=3,
	optim="adamw_torch",
	metric_for_best_model="wer_loss",
	predict_with_generate=True,
	load_best_model_at_end = True,
	greater_is_better=False,
	fp16=False,
	num_train_epochs=25,
	save_only_model=True,
	#warmup_ratio=0.1
	)


early_stop = EarlyStoppingCallback(5, 0.0001)

assert(args.early_stop in [True, False])

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



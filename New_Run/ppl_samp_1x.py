'''
Version 1 :
- Manual Word Error Rate (WER) probability ratio for Out of Vocab (OOV) term substitutions.
- The regular span substitution follows the true probability distribution observed from the ground-truth data.


** TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1 python ppl_samp_1x.py ../../Data/LT/train_matrix_fd_man_aws.p LT/train/train ../../Data/LT/trgTrain.txt ../../Data/LT/srcTrain.txt /home/avadehra/scribendi/dwnldModel/gpt2/	- D6-1


**TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=2 python ppl_samp_1x.py ../../Data/LT/train_matrix_fd_aws_man.p LT/dev/dev /home/avadehra/scribendi/dataAudio/parsed/LIB_TED/dev_trg.txt /home/avadehra/scribendi/dataAudio/parsed/LIB_TED/dev_src0.txt /home/avadehra/scribendi/dwnldModel/gpt2/	- D6-2

'''

import pickle as p
import json
import subprocess
import multiprocessing as mp
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import re
import gc
from jiwer import wer
from thefuzz import fuzz
import random
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics.text import Perplexity
from torch.nn import CrossEntropyLoss
from collections import Counter


#Global dictionary to save precomputes text spans and their perplexity. So doesnt have to be recalculated.
sent_ppl_score = {}

def load_gpt2_model(gpt2_model_path, device=None):
	"""
	Load GPT2 model + tokenizer from a given path.
	"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = AutoModelForCausalLM.from_pretrained(gpt2_model_path)
	tokenizer = AutoTokenizer.from_pretrained(gpt2_model_path)
	tokenizer.pad_token_id = tokenizer.eos_token_id
	model = model.to(device)
	model.eval()
	return model, tokenizer, device


def calculate_perplexity(ip_text, model, tokenizer, device):
	"""
	Compute perplexity of a text using GPT-2.
	"""
	#Separate the texts which are already computed:
	global sent_ppl_score
	new_ip_text = []
	existing_ip_text = []
	existing_ppl_scores = []
	for t in ip_text:
		if t in sent_ppl_score:
			existing_ip_text.append(t)
			existing_ppl_scores.append(sent_ppl_score[t])
		else:
			new_ip_text.append(t)
	if new_ip_text:
		new_ppl_scores = []
		encoded_inputs = tokenizer(new_ip_text, padding=True, truncation=True, max_length=60, return_tensors="pt", add_special_tokens=False).to(device)
		with torch.no_grad():
			outputs = model(**encoded_inputs)
		logits = outputs.logits
		# shift for causal LM
		ppl_metric = Perplexity(ignore_index=tokenizer.pad_token_id)
		#ppl_scores = []
		for i in range(len(new_ip_text)):
			score = ppl_metric(preds=logits[i, :-1].unsqueeze(0).to('cpu'), target=encoded_inputs['input_ids'][i, 1:].unsqueeze(0).to('cpu'))
			new_ppl_scores.append(score.item())
			sent_ppl_score[new_ip_text[i]] = score.item()
		assert len(new_ppl_scores) == len(new_ip_text)
		#Combine the scores:
		ppl_scores = existing_ppl_scores + new_ppl_scores
		existing_ip_text = existing_ip_text + new_ip_text
	else:
		ppl_scores = existing_ppl_scores
	assert(len(ppl_scores) == len(ip_text))
	assert(len(existing_ip_text) == len(ip_text))
	lowest_ppl_idx = np.argmin(ppl_scores).item()
	return existing_ip_text[lowest_ppl_idx]


def closeMatch(word, freqDict):
	#fdk = list(freqDict.keys())
	ratioD = [[fuzz.ratio(word, i), i] for i in freqDict]
	ratioD.sort(reverse=True)
	return ratioD[0][1]


def partition_text(x, fd, p_threshold):
	if not type(x) is list:
		x = x.split()
	#OOV substititutions over the text - x
	if OOV_correction:
		x_new = []
		for i in x:
			if not i in fd:#OOV word
				rndmSmp = random.random() <= werProb
				if rndmSmp:
					replWord = closeMatch(i, fd)
				else:
					replWord = i
				x_new.append(replWord)
			else:
				x_new.append(i)
		x_new = " ".join(x_new)
		x_new = re.sub(r"\s+", " ", x_new).lstrip().rstrip()
		x = x_new.split()
	#Generate the partitions:
	partitions = [[x[0]]]#first partition contains the first word.
	if x[0] in fd:
		prev_in = 1
	else:
		prev_in = 0
	for i in range(1,len(x)):
		w_i = x[i]
		if prev_in == 0:
			if w_i in fd:
				prev_in = 1
			else:
				prev_in = 0
			partitions.append([w_i])
		else:
			if not w_i in fd:
				partitions.append([w_i])
				prev_in = 0
			else:
				#prev_in = 1, w_i in fd
				prev_frag = " ".join(partitions[-1]).lstrip().rstrip()
				current_frag = prev_frag + " " + w_i
				freq_prev = fd[prev_frag]
				if not current_frag in fd:
					freq_curr = 0
				else:
					freq_curr = fd[current_frag]
				freq_phi = freq_curr/freq_prev
				#assert(0.0 <= freq_phi <= 1.0), "Value : "+str(freq_phi)
				if freq_phi >= p_threshold:#0.50:
					#Part of the previous fragment:
					partitions[-1].append(w_i)
					#prev_in = 1
				else:
					partitions.append([w_i])
				prev_in = 1
	part_lens = []
	for p in partitions:
		#p = " ".join(p)
		part_lens.append(len(p))
	#part_lens = str(Counter(part_lens))
	return partitions, part_lens

def sample_replacement(frag, m):
	#OOV - frag as it is right now:
	if not frag in m:
		return frag
	else:
		repl = []
		repl_freq = []
		for i in m[frag]:
			repl.append(i)
			repl_freq.append(m[frag][i][1])
		tmp = random.choices(repl, repl_freq)[0]
		return tmp



def best_substitution(prefix_tokens, target_tokens, candidates, max_n=6):
	"""
	Select best substitution based on GPT2 perplexity with max_n truncation.
	
	prefix_tokens : list of tokens before the target span
	target_tokens : list of tokens to be replaced
	candidates : list of possible candidates for substitution
	max_n : maximum length of prefix+candidate to consider (like n-gram limit)
	"""
	choice_dict = {}
	possible_choices = []
	for c in candidates:
		c = c.strip().split()
		new_sent = prefix_tokens + c
		new_sent = new_sent[-max_n:]
		new_sent = " ".join(new_sent).strip()
		new_sent = re.sub(r"\s+", " ", new_sent).strip()
		possible_choices.append(new_sent)
		choice_dict[new_sent] = c
	#calculate_perplexity(ip_text, model, tokenizer, device):
	best_span = calculate_perplexity(possible_choices, model, tokenizer, device)
	best_choice = choice_dict[best_span]#candidates[best_idx]
	return best_choice#.strip().split()

def get_corrpt_sent(p_thresh):
	L = len(src_trg_pair)
	srcSENTS = []
	trgSENTS = []
	editedSENTS = []
	partition_lens = []
	for i in tqdm(range(L)):
		s = src_trg_pair[i][0].strip()
		t = src_trg_pair[i][1].strip()
		if s and t:
			tmpEdit = []
			iPart, p_len = partition_text(s, fdM, p_thresh)
			for j in range(len(iPart)):
				#Partition at a time
				p = iPart[j]
				p_str = " ".join(p).strip()
				if not p_str in fdM:
					tmpEdit += p
				else:
					p_cand_vals = mM[p_str]
					p_cands = list(p_cand_vals.keys())
					#Add p_str as a candidate:
					p_cands.append(p_str)
					p_cands = list(set(p_cands))
					if (len(p_cands) == 1):
						p_edit = p_cands[0].strip().split()
					else:
						#def best_substitution(prefix_tokens, target_tokens, candidates, max_n=6):
						p_edit = best_substitution(prefix_tokens = tmpEdit[-6:], target_tokens = p, candidates = p_cands, max_n=5)
					#assert not p_edit is None
					assert(p_edit != None)#, "prefix: "+ str(tmpEdit[-7:]) + " target: "+ str(p) + " cands: "+ str(p_cands)
					tmpEdit += p_edit
			if tmpEdit:
				srcSENTS.append(s)
				trgSENTS.append(t)
				tmpEdit = " ".join(tmpEdit)
				tmpEdit = re.sub(r"\s+"," ",tmpEdit).strip()
				editedSENTS.append(tmpEdit)
				partition_lens += p_len
	#Write Out the Infornation and calculate the results of WER:
	partition_lens = str(Counter(partition_lens))
	thresh_val = str(int(p_thresh * 100))
	x = open(opLoc+ "_" + edit_type + "_" + thresh_val + ".txt", "w")#xyz/td_dev_corrupt_05.txt
	tmpwer = wer(trgSENTS, editedSENTS)
	#print("WER Probability: ", werProb)
	wO = x.write(str(round(tmpwer,6))+"\n\n")
	wO = x.write(partition_lens+"\n\n")
	#Printout the length of the source, target and edited sentences:
	wO = x.write("Set Length: "+str(len(trgSENTS))+"\n")
	x.close()




if __name__ == '__main__':
	'''
	Script Input:
	1 - Matrix Location
	2 - Output Location + file type.
	3 - Ground Truth Target
	4 - Ground Truth Source
	5 - location of the GPT2 model.
	'''
	mat_loc = sys.argv[1]#"sent_pair/train_matrix_fd_man_aws.p"
	# if man_aws then type == "corrupt", if aws_man then type == "correct".
	assert ("train_matrix_fd_" in mat_loc)
	mat_edit = mat_loc.split("train_matrix_fd_")[-1].split(".")[0]
	if mat_edit == "man_aws":#TRG -> SRC
		edit_type = "corrupt"
	elif mat_edit == "aws_man":#SRC -> TRG
		edit_type = "correct"
	else:
		raise ValueError("Matrix name should contain either `man_aws` or `aws_man`")
		sys.exit(1)
	opLoc = sys.argv[2]#fLoc+"corrupt_en/"
	assert("/" in opLoc)
	#Load and read the Source and Target .txt file and calculate the WER.
	gt_trg = sys.argv[3]#fLoc+"de_proc_transc.p"
	gt_src = sys.argv[4]#fLoc+"en_proc_transc.p"
	gt_trg = open(gt_trg, "r").read().strip().split("\n")
	gt_src = open(gt_src, "r").read().strip().split("\n")
	assert (len(gt_trg) == len(gt_src))
	src_trg_pair = {}
	for i in range(len(gt_trg)):
		if edit_type == "corrupt":#man_aws = TRG -> SRC
			src_trg_pair[i] = [gt_trg[i], gt_src[i]]
		elif edit_type == "correct":#aws_man = SRC -> TRG
			src_trg_pair[i] = [gt_src[i], gt_trg[i]]
	tmp = p.load(open(mat_loc, "rb"))
	mM = tmp[0]
	fdM = tmp[1]
	#GP2T2 model location:
	gpt2_path = sys.argv[5]
	model, tokenizer, device = load_gpt2_model(gpt2_path)
	OOV_correction = True
	werProb = wer(gt_trg, gt_src)
	x = open(opLoc+ "_" + edit_type + "_orig.txt", "w")
	print("WER Probability: ", werProb)
	wO = x.write(str(round(werProb,6)))
	x.close()
	del gt_trg, gt_src
	gc.collect()
	prob_threshold = np.linspace(0, 1, 21)[1:-1]
	#Non-multiprocessing the get_corrpt_sent function:
	for p_thresh in tqdm(prob_threshold):
		get_corrpt_sent(p_thresh)
	#Multiprocessing the get_corrpt_sent function:
	#cCount = mp.cpu_count() - 2
	#with mp.Pool(cCount) as p:
	#	r = list(tqdm(p.imap(get_corrpt_sent,prob_threshold), total = len(prob_threshold)))
'''
Version 2 :
- Manual Word Error Rate (WER) probability ratio for Out of Vocab (OOV) term substitutions.
- Manual WER for span substitutions. So, better corruptions can be generated.
- WER of 40.0 after substitutions, generates corrupted transcript with WER = 41.20
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
#from text_to_num import alpha2digit
from thefuzz import fuzz
import random
import sys

'''
Python script to use the error-matrix generated from the ground-truth data to incorporate pseudo corruptions/corrections in the 50k English AWS Processed Trancscripts. We use multiprocessing to run the script and save the individual corruptions/corrections.


** NOTE:
As described in the error generation project : `data_processing/Ground_Truth_DataProc_English/error_matrix_mapping.py` the ground truth data is used to learn 2 variations of the span substitutions:
- Manual -> AWS (pseudo corruption data)
- AWS -> Manual (pseudo correction data)

Depending on the the matrix+frequency_distribution data, the AWS 50k calls can be used as a seed for generating - corrections or corruptions!

** Example Usage:
>>> python error_substitutions.py [matrix_location] [en_proc_trnsc_location] [opLocation]

- [matrix_location] : Location for the `train_matrix_fd_man_aws.p` or `train_matrix_fd_aws_man.p` matrix and frequency distribution file.
- [en_proc_trnsc_location] : Location for the cleaned and processed AWS 50k transcripts `en_proc_transc_anon.p`
- [opLocation] : The location to save the individual error-substituted transcripts.
'''

def closeMatch(word, freqDict):
	fdk = list(freqDict.keys())
	ratioD = [fuzz.ratio(word, i) for i in fdk]
	maxR = max(ratioD)
	maxI = ratioD.index(maxR)
	return fdk[maxI]

def partition_text(x, fd):
	if not type(x) is list:
		x = x.split()
	#OOV substititutions over the text - x
	if OOV_correction:
		x_new = []
		for i in x:
			if not i in fd:#OOV word
				rndmSmp = random.random() <= oovProb
				if rndmSmp:
					replWord = closeMatch(i, fd)
				else:
					replWord = i
				x_new.append(replWord)
			else:
				x_new.append(i)
		x_new = " ".join(x_new)
		x_new = re.sub(r"\s+", " ", x_new)
		x = x_new.split()
	#w_1 in f_1
	partitions = [[x[0]]]
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
				prev_frag = " ".join(partitions[-1])
				current_frag = prev_frag + " " + w_i
				freq_prev = fd[prev_frag]
				if not current_frag in fd:
					freq_curr = 0
				else:
					freq_curr = fd[current_frag]
				freq_phi = freq_curr/freq_prev
				if freq_phi >= 0.50:
					#Part of the previous fragment:
					partitions[-1].append(w_i)
					#prev_in = 1
				else:
					partitions.append([w_i])
				prev_in = 1
	partF = []
	for p in partitions:
		p = " ".join(p)
		partF.append(p)
	return partF

def sample_replacement(frag, m):
	#OOV - frag as it is right now:
	rndmSmp = random.random() > werProb
	if rndmSmp:#
		return frag
	elif not frag in m:
		return frag
	else:
		repl_prob = []
		repl_freq = []
		mK = list(m[frag].keys())
		mK = [j for j in mK if not j == frag]
		if len(mK) == 0:#if no alternatives except the same word-fragment!
			return frag
		for i in mK:
			repl_freq.append(m[frag][i][0])
		for i in repl_freq:
			p = i/sum(repl_freq)
			repl_prob.append(p)
		tmp = random.choices(mK, repl_prob)[0]
		return tmp

def get_corrpt_sent(kID):
	# Create the noisy text:
	# Create aws -> corruption:
	trnscrpt = en_call_proc[kID]
	if trnscrpt:
		tmpA = []
		iPart = partition_text(trnscrpt, fdM)
		for j in iPart:
			repl = sample_replacement(j, mM)
			tmpA.append(repl)
		tmpA = " ".join(tmpA)
		tmpA = re.sub(r"\s+"," ",tmpA).strip()
	else:
		tmpA = ""
	x = open(opLoc+str(kID)+".txt", "w")
	tmp = x.write(tmpA)
	x.close()


if __name__ == '__main__':
	mat_loc = sys.argv[1]#"sent_pair/train_matrix_fd_man_aws.p"
	en_trnsc_loc = sys.argv[2]#fLoc+"en_proc_transc.p"
	opLoc = sys.argv[3]#fLoc+"corrupt_en/"
	#fLoc = "/datasets/ankitUW/resources/trnsc_op/"
	#dfUSP = pd.read_pickle(fLoc+"df_unsup.pd")
	tmp = p.load(open(mat_loc, "rb"))
	mM = tmp[0]
	fdM = tmp[1]
	OOV_correction = True
	oovProb = 0.159 * 2#Probability of OOV Words switching.
	werProb = 0.159 * 2#Probability of span substitutions.
	print("WER Probability: ", werProb)
	#en_call_proc = p.load(open(en_trnsc_loc, "rb"))
	en_tmp = open(en_trnsc_loc, "r").read().strip().split("\n")
	#convert into a dictionary with 0-N index - en_call_proc
	en_call_proc = {}
	for i in range(len(en_tmp)):
		en_call_proc[i] = en_tmp[i]
	en_id = list(en_call_proc.keys())
	cCount = mp.cpu_count() - 5
	with mp.Pool(cCount) as p:
		r = list(tqdm(p.imap(get_corrpt_sent,en_id), total = len(en_id)))
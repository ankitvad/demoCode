'''
Version 1 :
- Manual Word Error Rate (WER) probability ratio for Out of Vocab (OOV) term substitutions.
- The regular span substitution follows the true probability distribution observed from the ground-truth data.
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

'''
Python script to use the error-matrix generated from the ground-truth data to incorporate pseudo corruptions/corrections in the Processed Trancscripts. We use multiprocessing to run the script and save the individual corruptions/corrections.

** NOTE:
** Example Usage:
>>> python error_substitutions.py [matrix_location] [en_proc_trnsc_location] [opLocation]

- [matrix_location] : Location for the `train_matrix_fd_man_aws.p` or `train_matrix_fd_aws_man.p` matrix and frequency distribution file.
- [en_proc_trnsc_location] : Location for the cleaned and processed transcripts - `en_proc_transc_anon.p`
- [opLocation] : The location to save the individual error-substituted transcripts.
'''

def closeMatch(word, freqDict):
	#fdk = list(freqDict.keys())
	ratioD = [[fuzz.ratio(word, i), i] for i in freqDict]
	ratioD.sort(reverse=True)
	return ratioD[0][1]


def partition_text(x, fd):
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

def get_corrpt_sent(kID):
	# Create the noisy text:
	# Create aws -> corruption:
	trnscrpt = en_call_proc[kID]
	tmpA = []
	iPart = partition_text(trnscrpt, fdM)
	for j in iPart:
		repl = sample_replacement(j, mM)
		tmpA.append(repl)
	tmpA = " ".join(tmpA)
	tmpA = re.sub(r"\s+"," ",tmpA)
	x = open(opLoc+str(kID)+".txt", "w")
	tmp = x.write(tmpA)
	x.close()

if __name__ == '__main__':
	'''
	Script Input:
	1 - Matrix Location
	2 - .txt file with unsupervised transcripts to convert.
	3 - Output Location
	4 - Ground Truth Target
	5 - Ground Truth Source
	'''
	mat_loc = sys.argv[1]#"sent_pair/train_matrix_fd_man_aws.p"
	en_trnsc_loc = sys.argv[2]#fLoc+"en_proc_transc.p"
	opLoc = sys.argv[3]#fLoc+"corrupt_en/"
	#Load and read the Source and Target .txt file and calculate the WER.
	gt_trg = sys.argv[4]#fLoc+"de_proc_transc.p"
	gt_src = sys.argv[5]#fLoc+"en_proc_transc.p"
	gt_trg = open(gt_trg, "r").read().strip().split("\n")
	gt_src = open(gt_src, "r").read().strip().split("\n")
	tmp = p.load(open(mat_loc, "rb"))
	mM = tmp[0]
	fdM = tmp[1]
	OOV_correction = True
	werProb = wer(gt_trg, gt_src)
	print("WER Probability: ", werProb)
	del gt_trg, gt_src
	gc.collect()
	#en_call_proc = p.load(open(en_trnsc_loc, "rb"))
	en_tmp = open(en_trnsc_loc, "r").read().strip().split("\n")
	#convert into a dictionary with 0-N index - en_call_proc
	en_call_proc = {}
	for i in range(len(en_tmp)):
		en_call_proc[i] = en_tmp[i]
	en_id = list(en_call_proc.keys())
	cCount = mp.cpu_count() - 2
	with mp.Pool(cCount) as p:
		r = list(tqdm(p.imap(get_corrpt_sent,en_id), total = len(en_id)))
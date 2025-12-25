
import pickle as pkl
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
from rapidfuzz import distance
import random
import os
import sys
from collections import Counter
from difflib import get_close_matches
from metaphone import doublemetaphone


'''
Python script to use the error-matrix generated from the ground-truth data to incorporate pseudo corruptions/corrections in the Processed Trancscripts. We use multiprocessing to run the script and save the individual corruptions/corrections.

** NOTE:
** Example Usage:
>>> python error_substitutions.py [matrix_location] [en_proc_trnsc_location] [opLocation]

- [matrix_location] : Location for the `train_matrix_fd_man_aws.p` or `train_matrix_fd_aws_man.p` matrix and frequency distribution file.
- [en_proc_trnsc_location] : Location for the cleaned and processed transcripts - `en_proc_transc_anon.p`
- [opLocation] : The location to save the individual error-substituted transcripts.
'''

def closeMatch_fuzz(word, freqDict):
	#fdk = list(freqDict.keys())
	ratioD = [[fuzz.ratio(word, i), i] for i in freqDict]
	ratioD.sort(reverse=True)
	return ratioD[0][1]

def closeMatch_dist(word, freqDict, pron_dict):
	#fdk = list(freqDict.keys())
	w = get_close_matches(word, list(freqDict.keys()), n=1, cutoff=0.50)
	if not w:
		tmp = closeMatch_fuzz(word, freqDict)
		return tmp
	return w[0]

def closeMatch_prefix(word, freqDict, pron_dict):
	mp = doublemetaphone(word)
	if mp[0] == "":
		tmp = closeMatch_fuzz(word, freqDict)
		return tmp
	#distance.Prefix.similarity
	#Find the closest match among the candidates:
	ratioD = [[distance.Prefix.normalized_similarity(mp[0], p), p] for p in pron_dict]
	ratioD.sort(reverse=True)
	if ratioD[0][0] == 0:
		#No match found. Return the word itself.
		tmp = closeMatch_fuzz(word, freqDict)
		return tmp
	candidates = pron_dict[ratioD[0][1]]
	#Candidates is a frequency dictionary. Sample from that.
	repl = []
	repl_freq = []
	for i in candidates:
		repl.append(i)
		repl_freq.append(candidates[i])
	tmp = random.choices(repl, repl_freq)[0]
	return tmp


def partition_text(x, fd, pron_dict):
	if not type(x) is list:
		x = x.split()
	#OOV substititutions over the text - x
	if OOV_correction:
		x_new = []
		for i in x:
			if not i in fd:#OOV word
				rndmSmp = random.random() <= werProb
				if rndmSmp:
					replWord = closeMatch(i, fd, pron_dict)
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
				if freq_phi >= p_threshold:
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
	iPart = partition_text(trnscrpt, fdM, pron_dict)
	for j in iPart:
		repl = sample_replacement(j, mM)
		tmpA.append(repl)
	tmpA = " ".join(tmpA)
	tmpA = re.sub(r"\s+"," ",tmpA)
	if mp_flag == "not-mp":
		return tmpA
	else:
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
	6 - OOV Vocabulary type - prefix or distance
	7 - Multiprocessing or not - mp/not-mp
	'''
	mat_loc = sys.argv[1]#"sent_pair/train_matrix_fd_man_aws.p"
	en_trnsc_loc = sys.argv[2]#fLoc+"en_proc_transc.p"
	opLoc = sys.argv[3]#fLoc+"corrupt_en/"
	#Load and read the Source and Target .txt file and calculate the WER.
	gt_trg = sys.argv[4]#fLoc+"de_proc_transc.p"
	gt_src = sys.argv[5]#fLoc+"en_proc_transc.p"
	oov_type = sys.argv[6]#"prefix"#"distance"
	mp_flag = sys.argv[7]#"mp"#"not-mp"
	assert mp_flag in ["mp", "not-mp"], "Please provide a valid multiprocessing flag - mp/not-mp"
	if oov_type == "prefix":
		closeMatch = closeMatch_prefix
	else:
		closeMatch = closeMatch_dist
	gt_trg = open(gt_trg, "r").read().strip().split("\n")
	gt_src = open(gt_src, "r").read().strip().split("\n")
	tmp = pkl.load(open(mat_loc, "rb"))
	mM = tmp[0]
	fdM = tmp[1]
	pron_dict = tmp[2]
	OOV_correction = True
	werProb = wer(gt_trg, gt_src)
	p_threshold = 0.05
	print("Threshold Probability: ", p_threshold)
	print("WER Probability: ", werProb)
	del gt_trg, gt_src
	gc.collect()
	#en_call_proc = pkl.load(open(en_trnsc_loc, "rb"))
	en_tmp = open(en_trnsc_loc, "r").read().strip().split("\n")
	#convert into a dictionary with 0-N index - en_call_proc
	en_call_proc = {}
	for i in range(len(en_tmp)):
		en_call_proc[i] = en_tmp[i]
	en_id = list(en_call_proc.keys())
	if mp_flag == "not-mp":
		mod_sent = []
		for kID in tqdm(en_id):
			sNew = get_corrpt_sent(kID)
			mod_sent.append([kID, sNew])
		pkl.dump(mod_sent, open(opLoc+"CORRUP.p", "wb"))
	else:
		cCount = mp.cpu_count() - 2
		with mp.Pool(cCount) as p:
			r = list(tqdm(p.imap(get_corrpt_sent,en_id), total = len(en_id)))
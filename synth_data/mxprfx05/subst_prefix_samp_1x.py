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
from collections import Counter
from difflib import get_close_matches
from metaphone import doublemetaphone
from rapidfuzz import distance

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
				#assert(0.0 <= freq_phi <= 1.0), "Value : "+str(freq_phi)
				if freq_phi >= p_threshold:#0.50:
					#Part of the previous fragment:
					partitions[-1].append(w_i)
					#prev_in = 1
				else:
					partitions.append([w_i])
				prev_in = 1
	#part_lens = []
	#for p in partitions:
		#p = " ".join(p)
	#	part_lens.append(len(p))
	#part_lens = str(Counter(part_lens))
	return partitions

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



def best_substitution(prefix_tokens, target_tokens, candidates, ngram_dict, max_n=6):
	"""
	prefix_tokens : combined list of prefix tokens in the sentence (e.g., ["a","b","c","d","e"])
	target_tokens   : list of tokens to be replaced. After prefix (e.g., ["f","g"])
	candidates      : list of candidate replacements (each a string of tokens)
	ngram_dict      : dict mapping ngram string -> count
	max_n           : max ngram order available in dictionary
	
	The target to be edited is always at the END of the sentence.
	"""
	#p_cand_vals = mM[p_str]
	p_cands = list(candidates.keys())
	best_candidate = None
	best_score = (-1, -1)# (ngram_prefix_length, freq)
	# Everything before the edit span
	#prefix = sentence_tokens[:-len(target_tokens)]# if target_tokens else sentence_tokens
	prefix = prefix_tokens
	for cand in p_cands:
		cand = cand.strip().split()
		new_sent = prefix + cand# replace target with candidate
		# available context length (cannot exceed sentence length)
		avail_n = min(len(new_sent), max_n)
		# search longest prefix ending at candidate
		for n in range(avail_n, len(cand)-1, -1):
			ngram = " ".join(new_sent[-n:]).strip() #tuple(new_sent[-n:]) - right-aligned ending at candidate
			if ngram in ngram_dict:
				freq = ngram_dict[ngram]
				score = (n - len(cand), freq)#prefix_len x freq
				if score[0] == best_score[0]:
					cand_pair = [best_candidate, cand]
					freq_pair = [best_score[1], freq]
					tmp = random.choices(cand_pair, freq_pair)[0]
					if tmp == cand:
						best_score = score# (n - len(cand), freq)
						best_candidate = cand
				elif score[0] > best_score[0]:
					best_score = score
					best_candidate = cand
				break# stop backing off for this candidates.
	if not best_candidate:
		cand_pair = []
		freq_pair = []
		for c in candidates:
			cand_pair.append(c.strip().split())
			freq_pair.append(candidates[c][0])#freq
		best_candidate = random.choices(cand_pair, freq_pair)[0]
	return best_candidate#, best_score



def get_corrpt_sent(idx):
	s = en_call_proc[idx].strip()
	#p_thresh = p_threshold
	tmpEdit = []
	iPart = partition_text(s, fdM, pron_dict)
	for j in range(len(iPart)):
		#Partition at a time
		p = iPart[j]
		p_str = " ".join(p).strip()
		if not p_str in fdM:
			tmpEdit += p
		else:
			p_cand_vals = mM[p_str]
			p_cands = list(p_cand_vals.keys())
			if (len(p_cands) == 1):
				p_edit = p_cands[0].strip().split()
			else:
				p_edit = best_substitution(prefix_tokens = tmpEdit[-7:], target_tokens = p, candidates = p_cand_vals, ngram_dict = fdM2, max_n=6)
			#assert not p_edit is None
			assert(p_edit != None)#, "prefix: "+ str(tmpEdit[-7:]) + " target: "+ str(p) + " cands: "+ str(p_cands)
			tmpEdit += p_edit
	#Combine The string and write it out.
	tmpA = " ".join(tmpEdit)
	tmpA = re.sub(r"\s+"," ",tmpA).strip()
	if mp_flag == "not-mp":
		return tmpA
	else:
		x = open(opLoc+str(idx)+".txt", "w")
		tmp = x.write(tmpA)
		x.close()




if __name__ == '__main__':
	'''
	Script Input:
	1 - Matrix Location
	2 - Output Location + file type.
	3 - Ground Truth Target
	4 - Ground Truth Source
	5 - Other Transformation Matrix.
	6 - .txt file with unsupervised transcripts to convert.
	7- OOV Vocabulary type - prefix or distance
	8 - Multiprocessing or not - mp/not-mp
	'''
	mat_loc = sys.argv[1]#"sent_pair/train_matrix_fd_man_aws.p"
	# if man_aws then type == "corrupt", if aws_man then type == "correct".
	opLoc = sys.argv[2]#fLoc+"corrupt_en/"
	#Load and read the Source and Target .txt file and calculate the WER.
	gt_trg = sys.argv[3]#fLoc+"de_proc_transc.p"
	gt_src = sys.argv[4]#fLoc+"en_proc_transc.p"
	gt_trg = open(gt_trg, "r").read().strip().split("\n")
	gt_src = open(gt_src, "r").read().strip().split("\n")
	assert (len(gt_trg) == len(gt_src))
	tmp = pkl.load(open(mat_loc, "rb"))
	mM = tmp[0]
	fdM = tmp[1]
	pron_dict = tmp[2]
	mat_transf = sys.argv[5]
	assert(mat_transf != mat_loc), "Transformation matrices should be different."
	#Load the other transformation matrix:
	tmp2 = pkl.load(open(mat_transf, "rb"))
	_ = tmp2[0]
	fdM2 = tmp2[1]
	_ = tmp2[2]
	OOV_correction = True
	werProb = wer(gt_trg, gt_src)
	p_threshold = 0.05
	print("Threshold Probability: ", p_threshold)
	print("WER Probability: ", werProb)
	en_trnsc_loc = sys.argv[6]#fLoc+"en_proc_transc.p"
	en_tmp = open(en_trnsc_loc, "r").read().strip().split("\n")
	#convert into a dictionary with 0-N index - en_call_proc
	en_call_proc = {}
	for i in range(len(en_tmp)):
		en_call_proc[i] = en_tmp[i]
	en_id = list(en_call_proc.keys())
	del gt_trg, gt_src, en_tmp
	gc.collect()
	oov_type = sys.argv[7]#"prefix"#"distance"
	if oov_type == "prefix":
		closeMatch = closeMatch_prefix
	else:
		closeMatch = closeMatch_dist
	mp_flag = sys.argv[8]#"mp"#"not-mp"
	assert mp_flag in ["mp", "not-mp"], "Please provide a valid multiprocessing flag - mp/not-mp"
	#Non-multiprocessing the get_corrpt_sent
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
#File to take a list of lists with Reference and Hypothesis pairs and calculates the Word Error Rate for all the R,H pairs and saves it in a text file.
#The file uses multiprocessing to handle all pairs of R,H in parallel.

from jiwer import wer
import sys
import subprocess
from tqdm import tqdm
import glob
import os, pathlib
import gc
import pickle as p
import multiprocessing as mp


def calc_wer(ref_hyp_pair):
	ref, hyp, opFile = ref_hyp_pair
	#Open and read both files
	ref = open(ref, "r").read().strip().split("\n")
	hyp = open(hyp, "r").read().strip().split("\n")
	r_tmp = []
	h_tmp = []
	assert len(ref) == len(hyp), "Reference and hypothesis must have the same length."
	L = len(ref)
	#if R and H not null.
	for i in range(L):
		if ref[i] and hyp[i]:
			r_tmp.append(ref[i])
			h_tmp.append(hyp[i])
	w_score = wer(r_tmp, h_tmp)
	#Write out the score in - opLoc
	wOut = open(opLoc + opFile + ".txt", "w")
	tmp = wOut.write(str(w_score) + "\n")
	wOut.close()
	return 1

all_R_H_pairs = []

main_path = "/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/*/synthData/*/"
corrup_loc = glob.glob(main_path + "CORRUP")
correc_loc = glob.glob(main_path + "CORREC")

for c in corrup_loc:
	dsTyp = c.split("/")[7]
	assert dsTyp in ["RED", "LT"]
	dsTyp = dsTyp + "-" + c.split("/")[-2]
	r_ni = c.replace("/CORRUP", "/R_NI")
	h_ni = c.replace("/CORRUP", "/H_NI")
	all_R_H_pairs.append([r_ni, h_ni, dsTyp+"-NI"])
	#corrup = h, ASR = r
	all_R_H_pairs.append([c.split("corrup_")[0]+"ASR.txt", c, dsTyp])


for c in correc_loc:
	dsTyp = c.split("/")[7]
	assert dsTyp in ["RED", "LT"]
	dsTyp = dsTyp + "-" + c.split("/")[-2]
	r_ni = c.replace("/CORREC", "/R_NI")
	h_ni = c.replace("/CORREC", "/H_NI")
	all_R_H_pairs.append([r_ni, h_ni, dsTyp+"-NI"])
	#correc = r, ASR = h
	all_R_H_pairs.append([c, c.split("correc_")[0]+"ASR.txt", dsTyp])


opLoc = sys.argv[1]
cCount = mp.cpu_count() - 2

with mp.Pool(cCount) as p:
	r = list(tqdm(p.imap(calc_wer, all_R_H_pairs), total = len(all_R_H_pairs)))
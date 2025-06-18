#Function to ignore all insertion errors from hypothesis -> reference. Generate reference_tmp and return that sentence.
# Also test the generated sentence does not have any insert errors anymore!

import jiwer
import glob
import sys
import argparse
from tqdm import tqdm
import concurrent.futures
import time
import os, pathlib
import gc
import pickle as p
import multiprocessing as mp

#ref = MO, hyp = SRC
def remove_insert_edits(ref, hyp, do_test=True):
	out = jiwer.process_words(hyp, ref)#ref,hyp
	oA = out.alignments[0]
	ref_tmp = []
	r_s = ref.split()
	h_s = hyp.split()
	for i in oA:
		if i.type == "delete":
			#if it has to be deleted from hypothesis - that's it - not gotta do anything just ignore.
			continue
		elif i.type == "insert":
			#If something has to be inserted into hypothesis, again just ignore pick up from the next slot.
			continue
		elif i.type == "equal":
			ref_tmp.append(" ".join(h_s[i.ref_start_idx:i.ref_end_idx]))
		elif i.type == "substitute":
			ref_tmp.append(" ".join(r_s[i.hyp_start_idx:i.hyp_end_idx]))
	ref_tmp = " ".join(ref_tmp)
	if do_test == False:
		return ref_tmp
	#ref_tmp
	#Check the generated sentence is correct:
	out = jiwer.process_words(ref_tmp, ref)#r,h
	oA = out.alignments[0]
	for i in oA:
		assert(i.type in ["insert", "equal"])
	out = jiwer.process_words(hyp,ref_tmp)#r,h
	oA = out.alignments[0]
	for i in oA:
		assert(i.type != "insert")
	return ref_tmp


def openFiles(ref, hyp):
	#Open the files and return the contents.
	ref_content = open(ref, "r").read().strip().split("\n")
	hyp_content = open(hyp, "r").read().strip().split("\n")
	assert(len(ref_content) == len(hyp_content))
	r_h_pair = []
	L = len(ref_content)
	for i in range(L):
		r = ref_content[i]
		h = hyp_content[i]
		#If r and h not null.
		if r == "" or h == "":
			continue
		new_ref = remove_insert_edits(r, h, do_test=False)
		if new_ref:
			r_h_pair.append([new_ref, h])
	#Print the comparison of length of original ref_content and new r_h_pair.
	print(f"Original ref_content length: {len(ref_content)}, New r_h_pair length: {len(r_h_pair)}")
	return r_h_pair


def proc_locs(loc):
	ref = loc+"REF"
	hyp = loc+"HYP"
	r_h_pair = openFiles(ref, hyp)
	opW = loc+"R_H_NI.tsv"
	writeOut = open(opW, "w")
	#Separate using \t||\t delimiter.
	L = len(r_h_pair)
	for i in range(L):
		new_ref = r_h_pair[i][0]
		hyp = r_h_pair[i][1]
		tmpWO = writeOut.write(new_ref+"\t||\t"+hyp+"\n")
	'''
	if editType == "corrupt":
		opW = loc+"SRC_NI"
	elif editType == "correct":
		opW = loc+"CORREC_NI"
	writeOut = open(opW, "w")
	for i in newRef:
		tmpWO = writeOut.write(i+"\n")
	'''
	writeOut.close()
	return True



if __name__ == "__main__":
	# test:
	'''
	r = "a b c d e f g h"
	h = "b c i j e f h l"
	tmp = remove_insert_edits(r,h)
	assert(tmp == 'b c d e f h')
	h = "a b k l e g i h r"
	tmp = remove_insert_edits(r,h)
	assert(tmp == 'a b c d e g h')
	'''
	fLoc = "/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/"
	# corr_loc = glob.glob(fLoc+"*/synthData/corrup_*/")
	#LT_LOC = glob.glob(fLoc+"LT/synthData/*_*/")
	#RED_LOC = glob.glob(fLoc+"RED/synthData/*_*/")
	all_locs = glob.glob(fLoc+"*_*/")
	'''
	for i in tqdm(all_locs):
		proc_locs(i)
	'''
	#Parallely process all files in all_locs
	cCount = mp.cpu_count() - 2
	with tqdm(total=len(all_locs)) as pbar:
		pbar.set_description("Processing remove insert edits")
		with concurrent.futures.ProcessPoolExecutor(max_workers=cCount) as executor:
			futures = []
			for folder in all_locs:
				futures.append(executor.submit(proc_locs, folder))
			for future in concurrent.futures.as_completed(futures):
				pbar.update(1)



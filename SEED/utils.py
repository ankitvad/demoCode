import glob
import jiwer
import numpy as np
import pickle as p
import pandas as pd
from tqdm import tqdm


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


def getAllEdits(man,aws):
	pEditsM = []
	L = len(man)
	for i in range(L):
		out = jiwer.process_words(man[i], aws[i])
		a = aws[i].split()
		m = man[i].split()
		newE = []
		for o in out.alignments[0]:
			if not o.type == "equal":
				if o.type == "substitute":
					assert(o.ref_start_idx != o.ref_end_idx)
					assert(o.hyp_start_idx != o.hyp_end_idx)
					tmp = [o.type, " ".join(a[o.hyp_start_idx:o.hyp_end_idx]), " ".join(m[o.ref_start_idx:o.ref_end_idx])]
				else:#deleteee
					if o.ref_start_idx == o.ref_end_idx:
						tmp = [o.type, " ".join(a[o.hyp_start_idx:o.hyp_end_idx]), ""]
					elif o.hyp_start_idx == o.hyp_end_idx:
						tmp = [o.type, "", " ".join(m[o.ref_start_idx:o.ref_end_idx])]
				tmp = "|".join(tmp)
				newE.append(tmp)
		pEditsM.append(newE)
	assert(L == len(pEditsM))
	return pEditsM


def sum_and_divide(list_of_pairs):
	sum_num = 0
	sum_den = 0
	for pair in list_of_pairs:
		sum_num += pair[0]
		sum_den += pair[1]
	if sum_den == 0:
		return 0.0
	return sum_num / sum_den


def getEvalScore(man, aws, hyp):
	assert(len(man) == len(aws) == len(hyp))
	editMA = getAllEdits(man,aws)
	editHA = getAllEdits(hyp,aws)
	p_correct = []
	p_ignore = []
	p_introd = []
	#wer_am = []
	#wer_hm = []
	assert(len(editMA) == len(editHA) == len(man))
	L = len(editMA)
	for i in range(L):
		a = aws[i]
		m = man[i]
		h = hyp[i]
		if editMA[i]:
			corr = set(editMA[i]).intersection(set(editHA[i]))
			corr = [len(corr),len(editMA[i])]
			ign = set(editMA[i]).difference(set(editHA[i]))
			ign = [len(ign),len(editMA[i])]
			inc = set(editHA[i]).difference(set(editMA[i]))
			inc = [len(inc),len(editMA[i])]
			p_correct.append(corr)
			p_ignore.append(ign)
			p_introd.append(inc)
		#outAM = jiwer.process_words(m, a)
		#outHM = jiwer.process_words(m, h)
	#results:
	wer_am = jiwer.wer(man, aws)
	wer_hm = jiwer.wer(man, hyp)
	p_correct = sum_and_divide(p_correct)
	p_ignore = sum_and_divide(p_ignore)
	p_introd = sum_and_divide(p_introd)
	return (p_correct, p_ignore, p_introd, wer_am, wer_hm)



'''
test_trg = open(test_trg_LOC, "r").read().strip().split("\n")
test_src = open(test_src_LOC, "r").read().strip().split("\n")
hyp = open(f, "r").read().strip().split("\n")
assert(len(hyp) == len(test_src) == len(test_trg))
#Get the evaluation score:
p_correct, p_ignore, p_introd, wer_am, wer_hm = getEvalScore(test_trg, test_src, hyp)
#Write out the results:
tmpName = nm
writeOut.write(tmpName + "\n")
writeOut.write(f"p_correct: {p_correct}, p_ignore: {p_ignore}, p_introd: {p_introd}, wer_am: {wer_am}, wer_hm: {wer_hm}\n\n")
#Remove the inserts from the hypothesis:
hyp_NI = [remove_insert_edits(hyp[i], test_src[i], do_test=False) for i in range(len(hyp))]
p_correct, p_ignore, p_introd, wer_am, wer_hm = getEvalScore(test_trg, test_src, hyp_NI)
tmpName += "_NI :"
writeOut.write(tmpName + "\n")
writeOut.write(f"p_correct: {p_correct}, p_ignore: {p_ignore}, p_introd: {p_introd}, wer_am: {wer_am}, wer_hm: {wer_hm}\n\n")
writeOut.close()
'''
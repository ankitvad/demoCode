#Take as input a file of source sentences and a correction dictionary of substitution candidates.
#Only make corrections to 1,2, and 3 length words who are always wrongly transcribed.

import sys
import pickle as p
from rapidfuzz import distance
from metaphone import doublemetaphone
from tqdm import tqdm

'''
Input:

1 - correction matrices location
2 - corruption matrices location
3 - output location
4 - source sentences location
'''

correc_mat = sys.argv[1]
corrup_mat = sys.argv[2]
write_loc = sys.argv[3]
src_loc = sys.argv[4]

N_GRAM_MAX = 3

assert("aws_man" in correc_mat)
assert("man_aws" in corrup_mat)
tmpCorrec = p.load(open(correc_mat, "rb"))
mM, fdM, pron_dict = tmpCorrec[0], tmpCorrec[1], tmpCorrec[2]
tmpCorrup = p.load(open(corrup_mat, "rb"))
mM_corrup, fdM_corrup, pron_dict_corrup = tmpCorrup[0], tmpCorrup[1], tmpCorrup[2]

#words and phrases never transcribed properly = Present in AWS but not in MAN.
wrong_words = fdM.keys() - fdM_corrup.keys()#aws - man

#for w in wrong_words:
#	assert(w not in mM[w])

wrong_words_filter = {}
for w in wrong_words:
	#if fdM[w] > 1:
	if len(w.split()) <= N_GRAM_MAX :#Only consider 1,2,3 word phrases.
		wrong_words_filter[w] = mM[w]

#Sort wrong words filter by the number of words in sentence.
wrong_words_sorted = sorted(wrong_words_filter.items(), key=lambda x: len(x[0].split()), reverse=True)

#Function to generate all possible sub-spans of words and phrases from a given sentence.
#Given a phrase of length "n", generate all sub-phrases of length 1 to n-1.
def generate_subspans(phrase):
	#Only return subspans which are of Maximum N_GRAM_MAX length.
	subspans = {}
	words = phrase.split()
	n = len(words)
	for length in range(1, min(N_GRAM_MAX+1, n+1)):
		for start in range(0, n - length + 1):
			subspan = " ".join(words[start:start + length])
			subspans[subspan] = True
	return subspans

def replace_words(sentence, wrong_phrase, replacement):
	#ensure that only whole word phrases are replaced and not sub-strings.
	#the wrong_phrase may span multiple words. so Basically replace a sublist with another list.
	#Change wrong_phrase.split() to replacement.split() inside sentence.split()
	sentence_words = sentence.split()
	wp_words = wrong_phrase.split()
	rp_words = replacement.split()
	n = len(wp_words)
	new_sentence_words = []
	i = 0
	while i < len(sentence_words):
		if sentence_words[i:i+n] == wp_words:
			new_sentence_words.extend(rp_words)
			i += n
		else:
			new_sentence_words.append(sentence_words[i])
			i += 1
	return " ".join(new_sentence_words).strip()

src_sents = open(src_loc, "r").read().strip().split("\n")

edited_sents = []
ignore_phrases = {}

for s in tqdm(src_sents):
	#Separate the sentence s into all possible n-grams from 1 to 3. Then searching is quicker!
	tmp_s = generate_subspans(s)
	sent_edit = s
	for wp in wrong_words_sorted:
		wrong_phrase = wp[0]
		candidates = wp[1]
		if wrong_phrase in ignore_phrases:
			continue
		if wrong_phrase in tmp_s:
			subspans = generate_subspans(wrong_phrase)
			for sub in subspans:
				ignore_phrases[sub] = True
			#Make the substitution with the highest frequency candidate.
			if len(candidates) == 1:
				repl = list(candidates.keys())[0]
			else:
				#Search the Metaphone similarity with the least distance.
				sim_score = [distance.Levenshtein.distance(doublemetaphone(wrong_phrase)[0], doublemetaphone(cand)[0]) for cand in candidates.keys()]
				min_index = sim_score.index(min(sim_score))
				repl = list(candidates.keys())[min_index]
			sent_edit = replace_words(s, wrong_phrase, repl)
			#tmp_s = tmp_s.replace(wrong_phrase, repl)
	edited_sents.append(sent_edit)
	ignore_phrases = {}

assert(len(edited_sents) == len(src_sents))
out_f = open(write_loc, "w")
for es in edited_sents:
	tmp = out_f.write(es+"\n")
out_f.close()
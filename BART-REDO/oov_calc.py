import sys
import pickle as p
from nltk.corpus import words
setofwords = set(words.words())

matrix_location = sys.argv[1]
asr_location = sys.argv[2]

tmp = p.load(open(matrix_location, "rb"))
mM = tmp[0]
fdM = tmp[1]

en_tmp = open(asr_location, "r").read().strip().split("\n")
#Combine the list of list into dictionary of word counts.

word_count = {}
for line in en_tmp:
	words = line.split()
	for word in words:
		if word not in word_count:
			word_count[word] = 0
		word_count[word] += 1


#Compare each word in word_count with the frequency dictionary fdM to find OOV words. Keep a count of unique OOV words and total OOV occurrences.

#Load Existing Unique Words.
oov_words = p.load(open("oov_words.p", "rb"))

oov_unique_count = 0
oov_total_count = 0

for word in word_count:
	if word not in fdM:
		if word in setofwords or word.capitalize() in setofwords:
			oov_unique_count += 1
			oov_total_count += word_count[word]
			oov_words[word] = ""

p.dump(oov_words, open("oov_words.p", "wb"))


print("Unique Dict OOV words: ", oov_unique_count)
print("Total Dict OOV occurrences: ", oov_total_count)
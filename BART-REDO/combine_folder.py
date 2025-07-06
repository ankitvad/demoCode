#LT/srcASR_dict.p
import sys
import subprocess
from tqdm import tqdm
import glob
import concurrent.futures
import time
import os, pathlib
import gc
import pickle as p
import multiprocessing as mp

def readSent(x):
	tmp = open(x, "r").read().strip()
	return tmp

#get the name of the folder location where files from 0.txt to N.txt are saved.
#Load all the files using multiprocessing and combine them together.
#Save the file in the location and file provided as input by the use.

fold_loc = sys.argv[1]#"/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/LT/synthData/"
opFile = sys.argv[2]#"/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/LT/synthData/correc_3x/"

allFiles = glob.glob(fold_loc+"*.txt")
L = len(allFiles)
#Ensure that files from 0.txt to L-1.txt exist in fold_loc.
assert(L > 0), "No files found in the specified folder: "+fold_loc
#convert to dictionary:
allFilesD = {}
for i in allFiles:
	num_nm = int(i.split("/")[-1].split(".")[0])
	assert(num_nm not in allFilesD), "Duplicate file found: "+i
	allFilesD[num_nm] = i

#Check all file exists:
for i in range(L):
	assert(i in allFilesD), "File not found: "+fold_loc+str(i)+".txt"


with tqdm(total=L) as pbar:
	pbar.set_description("Processing sentences")
	with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count() - 5) as executor:
		# Submit tasks to the executor
		#these_futures = [executor.submit(gen_sents, ii[::-1]) for ii in sentA_M]#Pass - [M,A] -> R, H
		these_futures = {}
		for ii in range(L):
			locTmp = allFilesD[ii]#fold_loc + str(ii) + ".txt"
			#assert(locTmp in allFilesD), "File not found: "+locTmp
			these_futures[executor.submit(readSent, locTmp)] = ii
		results = {}
		for future in concurrent.futures.as_completed(these_futures):
			arg = these_futures[future]
			results[arg] = future.result()
			pbar.update(1)
		#concurrent.futures.wait(these_futures)

wLoc = fold_loc + opFile

wOut = open(wLoc, "w")
for jj in range(L):
	tmp = wOut.write(results[jj]+"\n")
wOut.close()
print("Saved combined file to: ", wLoc)
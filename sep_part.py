import sys
from itertools import permutations,combinations
import jiwer
from jiwer import wer
import numpy as np
import pickle as p
import re
import random
import subprocess
import multiprocessing as mp
import time
import pandas as pd
from tqdm import tqdm
import glob
import concurrent.futures
import time


'''
This script generates sentence pairs from a given dataset by processing the reference and hypothesis sentences, extracting equal alignments, and creating pairs based on specified conditions. It uses multiprocessing to handle large datasets efficiently.

Updated to work on extracted .txt files and not really on pandas dataframe.

python sep_partition.py [aws_transcripts-pickle] [corrup/correc_folder]

python sep_part.py /datasets/ankitUW/redoBART/Data/synthData/en_proc_transc_anon_6300.p /datasets/ankitUW/redoBART/Data/tmpSynth/
'''

def sent_pair(ref, hyp):
    out = jiwer.process_words(ref, hyp)
    oA = out.alignments[0]
    eqIndx = []
    for o in oA:
        if o.type =="equal":
            tmp = [o.ref_start_idx, o.ref_end_idx, o.hyp_start_idx, o.hyp_end_idx]
            eqIndx.append(tmp)
    eqSnt = []
    eqIndxCnt = len(eqIndx)
    eqIndxCnt = [jkl for jkl in range(eqIndxCnt)]
    eqPair = [e for e in combinations(eqIndxCnt,2)]
    eqPairDict = {}
    for jkl in eqPair:
        if not jkl[0] in eqPairDict:
            eqPairDict[jkl[0]] = []
    for jkl in eqPair:
        eqPairDict[jkl[0]].append(jkl)
    for jkl in eqPairDict:
        tmpJKL = eqPairDict[jkl]
        #print(tmpJKL)
        for mno in tmpJKL:
            assert(jkl == mno[0])
        tmpmno = tmpJKL[0][1]
        for mno in tmpJKL[1:]:
            assert(mno[1] > tmpmno)
            tmpmno = mno[1]
    for e in eqPair:
        e1 = eqIndx[e[0]]# 4 tuple
        e2 = eqIndx[e[1]]# 4 tuple
        eqSnt.append([e[0], e[1], e2[1] - e1[0]])
    return eqSnt, eqIndx

def s_dict(s):
    sDict = {}
    for i in s:
        if not i[0] in sDict:
            sDict[i[0]] = []
        sDict[i[0]].append(i)
    for j in sDict:
        tmp = sDict[j]
        tmpS = tmp[0][1]
        for k in tmp:
            assert(j == k[0])
            assert(tmpS <= k[1])
            tmpS = k[1]
    return sDict

def crt_prings(s):
    create_pairings = []
    startC = 0
    while startC < len(s):
        tmp = s[startC]
        sentSZ = random.randint(LB,UB)
        L = len(tmp)
        for i in range(L):
            if (tmp[-1][2] - tmp[i][2]) < sentSZ:
                create_pairings.append(tmp[-1])
                startC = tmp[-1][1]
                break
            if tmp[i+1][2] > sentSZ:
                create_pairings.append(tmp[i])
                startC = tmp[i][1]
                break
    #extracted_pairings.append(create_pairings)
    return create_pairings


def gen_sents(ipSent):#ref, hyp
    ref, hyp = ipSent
    r = ref.split()
    h = hyp.split()
    snt_pr, eq_al = sent_pair(ref, hyp)
    sDict = s_dict(snt_pr)
    create_pairings = crt_prings(sDict)
    #e_sents = extr_sent(create_pairings, eq_al)
    itemSent = []
    for j in create_pairings:
        e1 = eq_al[j[0]]#equal-alignment
        e2 = eq_al[j[1]]#equal-alignment
        refS = " ".join(r[e1[0]:e2[1]])
        hypS = " ".join(h[e1[2]:e2[3]])
        tmp = [refS,hypS]
        itemSent.append(tmp)
    return itemSent


if __name__ == '__main__':
    LB = 30
    UB = 40
    trnLoc = sys.argv[1]
    awsTrn = p.load(open(trnLoc, "rb"))
    ids = list(awsTrn.keys())
    subst_loc = sys.argv[2]
    correc_loc = glob.glob(subst_loc+"correc_*/")
    corrup_loc = glob.glob(subst_loc+"corrup_*/")
    A = []
    for i in ids:
        a = awsTrn[i]
        assert(a[-7:] == "dollars")
        a = a[:-7].strip()#remove dollars
        A.append(a)
    for c in correc_loc + corrup_loc:
        #For Correction:
        #AWS = SRC = HYP
        #MAN = REF = TRG
        #For Corruption:
        #AWS = TRG = REF
        #MAN = SRC = HYP
        sR = []
        sH = []
        sentA_M = []
        datType = c.split("/")[-2].split("_")[0]#correc/corrup
        assert(datType in ["correc", "corrup"])
        allMod = glob.glob(c+"*.txt")
        assert(len(allMod) == len(ids))
        #Load all the data to be chunked.
        M = []
        for i in ids:
            #a = awsTrn[i]
            #assert(a[-7:] == "dollars")
            #a = a[:-7].strip()#remove dollars
            m = open(c+i+".txt", "r").read().strip()
            if m[-7:] == "dollars":
                m = m[:-7].strip()
            M.append(m)
        assert(len(M) == len(A))
        sentA_M = list(zip(A, M))#[(A,M), (A,M), ...]
        #Parallel Process the generation.
        lF = len(sentA_M)
        with tqdm(total=lF) as pbar:
            pbar.set_description("Processing sentences")
            with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count() - 10) as executor:
                # Submit tasks to the executor
                #these_futures = [executor.submit(gen_sents, ii[::-1]) for ii in sentA_M]#Pass - [M,A] -> R, H
                cIdx = 0
                these_futures = {}
                for ii in sentA_M:
                    a,m = ii
                    if datType == "correc":
                        r = m
                        h = a
                    else:#corrupt
                        r = a
                        h = m
                    these_futures[executor.submit(gen_sents, [r,h])] = cIdx
                    cIdx += 1
                results = {}
                for future in concurrent.futures.as_completed(these_futures):
                    arg = these_futures[future]
                    results[arg] = future.result()
                    pbar.update(1)
            #concurrent.futures.wait(these_futures)
        for tmp in results:
            for j in results[tmp]:
                sR.append(j[0])
                sH.append(j[1])
        rO = open(c+"REF", "w")
        hO = open(c+"HYP", "w")
        assert(len(sR) == len(sH))
        L = len(sR)
        for k in range(L):
            tmpK = rO.write(sR[k]+"\n")
            tmpK = hO.write(sH[k]+"\n")
        rO.close()
        hO.close()
'''
['LT_Train_Dist_CORREC_GEN_SD.p', 'RED_Dev_PRFX_CORREC_GEN_SD.p', 'RED_Dev_Dist_CORREC_PARTITIONS.p', 'LT_Train_PRFX_CORREC_GEN_SD.p', 'FR_Dev_Dist_CORREC_PARTITIONS.p', 'RED_Dev_Dist_CORREC_GEN_SD.p', 'LT_Train_Dist_CORREC_PARTITIONS.p', 'FR_Dev_Dist_CORREC_GEN_SD.p', 'LT_Dev_PRFX_CORREC_GEN_SD.p', 'FR_Train_Dist_CORREC_PARTITIONS.p', 'LT_Dev_Dist_CORREC_PARTITIONS.p', 'FR_Train_Dist_CORREC_GEN_SD.p', 'RED_Train_PRFX_CORREC_GEN_SD.p', 'FR_Dev_PRFX_CORREC_GEN_SD.p', 'RED_Train_Dist_CORREC_PARTITIONS.p', 'LT_Dev_Dist_CORREC_GEN_SD.p', 'RED_Train_Dist_CORREC_GEN_SD.p', 'FR_Train_PRFX_CORREC_GEN_SD.p']
'''

import pickle as pkl
import glob


allLoc = {
"LT_Train_" : ["/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/LT/srcTrain.txt", "/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/LT/trgTrain.txt"],
 "RED_Dev_" : ["/home/avadehra/scribendi/dataAudio/parsed/red_src.txt", "/home/avadehra/scribendi/dataAudio/parsed/red_trg.txt"],
 "FR_Dev_" : ["/home/avadehra/scribendi/dataAudio/parsed/FR_DT_src.txt", "/home/avadehra/scribendi/dataAudio/parsed/FR_DT_trg.txt"],
 "LT_Dev_" : ["/home/avadehra/scribendi/dataAudio/parsed/lib_ted_src0.txt", "/home/avadehra/scribendi/dataAudio/parsed/lib_ted_trg.txt"],
 "FR_Train_" : ["/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/FR/srcTrain.txt", "/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/FR/trgTrain.txt"],
 "RED_Train_" : ["/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/RED/srcTrain.txt", "/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/RED/trgTrain.txt"]
}

#Create a mapping value. Load source and target in dictionary.
def load_text_file(path):
	tmp = open(path, "r").read().strip().split("\n")
	return tmp

all_mapping = {}

for key in allLoc.keys():
	srcPath = allLoc[key][0]
	trgPath = allLoc[key][1]
	assert("src" in srcPath), "Source path incorrect for key: {}".format(key)
	assert("trg" in trgPath), "Target path incorrect for key: {}".format(key)
	assert(srcPath != trgPath), "Source and Target paths are same for key: {}".format(key)
	srcLines = load_text_file(srcPath)
	trgLines = load_text_file(trgPath)
	assert(len(srcLines) == len(trgLines)), "Source and Target length mismatch for key: {}".format(key)
	mapping = {"source": srcLines, "target": trgLines}
	all_mapping[key] = mapping


pkl.dump(all_mapping, open("all_ST_data_mapping.pkl", "wb"))

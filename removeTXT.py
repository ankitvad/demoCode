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
import tarfile

#Combine all TXT files into a tar.gz file in each folder and then delete all the .txt files.
def compressTXT(folder):
	txt_files = glob.glob(os.path.join(folder, "*.txt"))
	if not txt_files:
		print(f"No .txt files found in {folder}")
		return
	tar_file_path = os.path.join(folder, "combined_files.tar.gz")
	with tarfile.open(tar_file_path, "w:gz") as tar:
		for txt_file in tqdm(txt_files):
			tar.add(txt_file, arcname=os.path.basename(txt_file))
	return True

def deleteTXT(folder):
	#Delete the files using pathlib.Path.unlink()
	txt_files = glob.glob(os.path.join(folder, "*.txt"))
	for txt_file in tqdm(txt_files):
		pathlib.Path(txt_file).unlink(missing_ok=True)
	return True


if __name__ == "__main__":
	fLoc = "/home/avadehra/scribendi/gotutiyan-BART/syntheticDataAudio/Data/"
	# corr_loc = glob.glob(fLoc+"*/synthData/corrup_*/")
	LT_LOC = glob.glob(fLoc+"LT/synthData/*_*/")
	RED_LOC = glob.glob(fLoc+"RED/synthData/*_*/")
	all_locs = LT_LOC + RED_LOC
	funcType = sys.argv[1]# "compress" or "delete"
	#Use multiprocessing to handle the jobs parallely.
	cCount = mp.cpu_count() - 2
	with tqdm(total=len(all_locs)) as pbar:
		pbar.set_description(f"Processing {funcType} operations")
		with concurrent.futures.ProcessPoolExecutor(max_workers=cCount) as executor:
			futures = []
			for folder in all_locs:
				if funcType == "compress":
					futures.append(executor.submit(compressTXT, folder))
				elif funcType == "delete":
					futures.append(executor.submit(deleteTXT, folder))
				else:
					raise ValueError("Unknown function type: " + funcType)
			for future in concurrent.futures.as_completed(futures):
				pbar.update(1)
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import numpy as np

def returnWER(i):
	tmp = float(open(i, "r").read().strip().split("\n")[0])
	return tmp

def print_graph(x,y):
	# Set figure size
	plt.figure(figsize=(6, 4))  
	# Create line chart
	sns.lineplot(x=x,y=y)
	plt.title("Threshold Plot")
	plt.xlabel("Threshold Range") # x-axis name
	plt.ylabel("WER Score") # y-axis name
	# Rotate x-axis labels to 45 degrees and reduce labels size
	plt.xticks(rotation=45, fontsize=7)
	# Display the plot
	plt.show()  

def returnParseVal(y):
	p_vals = []
	parse_dict = {}
	for i in y:
		if "_orig.txt" in i:
			parse_dict["orig"] = returnWER(i)
		else:
			nmVal = int(i.split("_")[-1].split(".")[0])
			parse_dict[nmVal] = returnWER(i)
			p_vals.append(nmVal)
	p_vals.sort()
	print("The Original WER : " + str(parse_dict["orig"]))
	wer_scores = [parse_dict[i] for i in p_vals]
	print_graph(p_vals, wer_scores)
	min_val = np.argmin(wer_scores)
	print("Best Threshold is : " + str(p_vals[min_val]))

def printOutMap(x):
	corrupt = glob.glob(x+"*corrupt*")
	correct = glob.glob(x+"*correct*")
	print("The Result for : Corruption - " + x)
	returnParseVal(corrupt)
	print("The Result for : Correction - " + x)
	returnParseVal(correct)

allLocs = glob.glob("*_scores/*/*/")
allLocs


printOutMap(allLocs[0])


import glob
import re
import string
import argparse


def fix_broken_unicode(text: str) -> str:
	# Step 1 — restore stripped unicode escapes: u00e9 → \u00e9
	text = re.sub(r'(?<!\\)u([0-9a-fA-F]{4})', r'\\u\1', text)
	# Step 2 — decode escaped sequences safely
	text = text.encode('latin1', errors='ignore').decode('unicode_escape', errors='ignore')
	# Step 3 — REMOVE illegal UTF-16 surrogates (the real crash cause)
	text = re.sub(r'[\ud800-\udfff]', '', text)
	return text

if __name__ == "__main__":
	#take the input location and output location
	#Read all files line by line and output the fixed version with -fixed tag.
	parser = argparse.ArgumentParser(description="Fix French Unicode Characters in Text Files")
	parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing text files')
	parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder for fixed text files')
	args = parser.parse_args()
	input_folder = args.input_folder.rstrip("/")
	output_folder = args.output_folder.rstrip("/")
	all_files = glob.glob(input_folder + "/*_FR_*.txt")
	for a in all_files:
		f_name = a.split("/")[-1]
		out_path = output_folder + "/" + f_name.replace(".txt", "-fixed.txt")
		with open(a, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8", errors="strict") as f_out:
			for line in f_in:
				fixed_line = fix_broken_unicode(line.strip())
				f_out.write(fixed_line + "\n")


import gc
import string
import re
import argparse
import json
from tqdm import tqdm
from typing import List, Tuple
#from rank_bm25 import BM25Okapi
import bm25s
import glob
import torch
import os
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from jiwer import wer
import pickle as pkl

def load_file(path: str) -> List[str]:
	tmp = pkl.load(open(path, "rb"))
	return tmp

def build_messages_joint(nbest_list_test: List[str], few_shot_examples: List[List], model_type: str):
	"""
	Consolidates instructions, examples, and task input into a single User message.
	Ensures compatibility with Mistral (no system role, no double user roles).
	"""
	# 1. Core Instructions
	instructions = (
		"You are an expert ASR (Automatic Speech Recognition) generated transcript correction assistant. "
		"Your task is to analyze a list of ASR hypotheses, considering all the possible variations "
		"and generate the single most accurate, final correct transcription. "
		"Do not translate the text into another Language. "
		"Generate the 'corrected_text' in a strictly valid JSON format."
	)
	messages = []
	# Use System Role for Llama/Qwen, otherwise prepend to User for Mistral
	if model_type != "mistral":
		messages.append({"role": "system", "content": instructions})
		user_content_parts = []
	else:
		user_content_parts = [instructions]
	# 2. Add Examples to the same content list if k > 0
	if few_shot_examples:
		user_content_parts.append("### Examples of List of Hypotheses and Corrections:")
		for ex in few_shot_examples:
			n_best = ex[0]
			corrected = ex[1]
			example_json = json.dumps({"hypotheses_list": n_best, "corrected_text": corrected}, indent=4)
			user_content_parts.append(example_json)
	# 3. Add the actual task input to the same content list
	task_input = json.dumps({"hypotheses_list": nbest_list_test},indent=4)
	user_content_parts.append(f"### Please provide the correction for the following Hypotheses:\n\n{task_input}")
	# Combine everything into ONE single User message
	messages.append({
		"role": "user",
		"content": "\n\n".join(user_content_parts) + "\n"
	})
	# 4. The Continuation Prompt (Assistant Role)
	# This remains separate as it is the "trigger" for the model to start responding.
	messages.append({
		"role": "assistant",
		"content": '\n{"corrected_text": "'
	})
	return messages

def normalize_text(text: str) -> str:
	"""
	Normalizes a sentence by:
	1. Converting to lower case.
	2. Removing all punctuation except apostrophes (').
	3. Collapsing multiple spaces into one.
	"""
	if not text:
		return ""
	# Convert to lower case
	text = text.lower()
	# Define punctuation to remove (all string.punctuation except ')
	# string.punctuation is: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
	punct_to_remove = string.punctuation.replace("'", "")
	# Use translation table for high-performance character removal
	table = str.maketrans("", "", punct_to_remove)
	text = text.translate(table)
	# Remove extra whitespace (optional but recommended for ASR)
	text = re.sub(r'\s+', ' ', text).strip()
	return text


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_folder", type=str, required=True, help="Folder containing all .p and .pkl files")
	parser.add_argument("--output_path", type=str, default="./")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--model_type", type=str, choices=['mistral', 'llama', 'qwen', 'deepseek'], required=True, help="Type of model to use")
	args = parser.parse_args()
	# Model Mapping
	#ALL_MODELS = ['mistral', 'llama', 'qwen', 'deepseek']
	ALL_MODELS = [args.model_type]
	K = [0,1,3]
	nlist_thresh = [0.05, 0.15, 0.25, 0.35]#, 0.50]
	model_map = {
		"mistral": "/model-weights/Mistral-7B-Instruct-v0.3/",
		"llama": "/model-weights/Meta-Llama-3.1-8B-Instruct/",
		"qwen": "/model-weights/Qwen2.5-7B-Instruct/",
		"deepseek": "/model-weights/DeepSeek-R1-Distill-Qwen-7B/"
	}
	#model_id = model_map[args.model_type]
	# Load Data
	test_files = glob.glob(os.path.join(args.input_folder, "*_Dev_*CORREC_GEN_SD.p"))
	st_mapping = load_file(os.path.join(args.input_folder, "all_ST_data_mapping.pkl"))
	for test_file in test_files:
		test_filename = test_file.split("/")[-1]
		print("Processing file: ", test_filename)
		# Determine dataset type from filename
		test_type = "_".join(test_filename.split("_")[:2])+"_"
		assert test_type in st_mapping, "Test type not found in mapping."
		dsType = test_type.split("_")[0]
		test_source = st_mapping[test_type]["source"]
		test_target = st_mapping[test_type]["target"]
		train_src = st_mapping[test_type.replace("Dev", "Train")]["source"]
		train_tgt = st_mapping[test_type.replace("Dev", "Train")]["target"]
		train_file = test_file.replace("_Dev_", "_Train_")
		test_hyp = load_file(test_file)
		train_hyp = load_file(train_file)
		for model_type in ALL_MODELS:
			model_id = model_map[model_type]
			tokenizer = AutoTokenizer.from_pretrained(model_id)
			llm = LLM(
				model=model_id,
				enable_prefix_caching = True,
				tensor_parallel_size = 1,
				max_model_len=4096*2,
				gpu_memory_utilization = 0.95
				)
			# We stop at the closing quote of the JSON field or the brace
			sampling_params = SamplingParams(
				temperature=0.1,
				#top_p=0.95,
				top_k=10,
				max_tokens=512,
				seed = args.seed,
				stop=["}",'"']
			)
			for k in K:
				bm25 = None
				if k > 0:
					tok_train_src = [line.split() for line in train_src]
					tok_test_src = [line.split() for line in test_source]
					bm25 = bm25s.BM25(corpus=train_src)
					bm25.index(tok_train_src)
					resultsIDX, _ = bm25.retrieve(tok_test_src, k=k, corpus = range(len(train_src)))
					assert(len(resultsIDX) == len(test_source))
					print("BM25 indexing completed.")
				# Initialize vLLM
				prompts = []
				#Use BM25S
				for iT in tqdm(range(len(test_source))):
					textS = test_source[iT]
					text_nlist = [test_hyp[n][iT][1] for n in nlist_thresh]
					assert(test_hyp[0.50][iT][0] == iT)
					text_nlist = text_nlist + [textS]
					few_shot = []
					if k > 0 and bm25:
						top_indices = resultsIDX[iT].tolist()
						few_shot = []
						for idx in top_indices:
							train_nlist = [train_hyp[n][idx][1] for n in nlist_thresh] + [train_src[idx]]
							few_shot.append([train_nlist, train_tgt[idx]])
					messages = build_messages_joint(text_nlist, few_shot, model_type)
					# Apply template using continue_final_message
					prompt = tokenizer.apply_chat_template(
						messages,
						tokenize=False,
						add_generation_prompt=False, # Must be False to allow the manual assistant start
						continue_final_message=True # Effectively 'picks up' from where our message left off
						)
					prompts.append(prompt)
				# Batch Generation
				outputs = llm.generate(prompts, sampling_params)
				# Final result construction
				final_output = []
				normalized_output = []
				for i, out in enumerate(outputs):
					# The model generated everything AFTER '{"corrected": "'
					correction_only = out.outputs[0].text.strip().rstrip('"').rstrip('}')
					final_output.append(correction_only)
					normalized_output.append(normalize_text(correction_only))
				# Write outputs
				op = {}
				op["outputs"] = final_output
				op["normalized_outputs"] = normalized_output
				opName = args.output_path + model_type +"_"+ test_filename.split("_CORREC_GEN_SD")[0] + "_" + str(k) + "_src_n-best"
				pkl.dump(op, open(opName+".pkl", "wb"))
				x = open(opName+".txt", "w", encoding="utf-8")
				for line in normalized_output:
					_ = x.write(line+"\n")
				x.close()
			# Clean up model parallel resources
			destroy_model_parallel()
			destroy_distributed_environment()
			#del llm.llm_engine.model_executor
			del llm
			gc.collect()
			torch.cuda.empty_cache()

if __name__ == "__main__":
	main()
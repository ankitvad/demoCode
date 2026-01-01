import string
import re
import argparse
import json
from tqdm import tqdm
from typing import List, Tuple
#from rank_bm25 import BM25Okapi
import bm25s
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from jiwer import wer
import pickle as pkl

def load_file(path: str) -> List[str]:
	tmp = open(path, "r", encoding="utf-8").read().strip().split("\n")
	return tmp

def build_messages_joint(input_text: str, few_shot_examples: List[Tuple[str, str]], k: int, model_type: str):
	"""
	Consolidates instructions, examples, and task input into a single User message.
	Ensures compatibility with Mistral (no system role, no double user roles).
	"""
	# 1. Core Instructions
	instructions = (
		"You are an expert ASR (Automatic Speech Recognition) generated transcript correction assistant. "
		"Your task is to check the 'asr_transcript' and fix any possible transcription issues like phonetic substitution errors, hallucinations, etc., while keeping the original meaning. "
		"Ignore punctuations, lower/upper case word issues, and don't change any words that seem correct. "
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
	if k > 0 and few_shot_examples:
		user_content_parts.append("### Below are some examples of ASR transcripts and the corrected text:")
		for src, tgt in few_shot_examples:
			example_json = json.dumps({"asr_transcript": src, "corrected_text": tgt}, indent=4)
			user_content_parts.append(example_json)
	# 3. Add the actual task input to the same content list
	task_input = json.dumps({"asr_transcript": input_text})
	user_content_parts.append(f"### Please provide the correction for the following input:\n\n{task_input}")
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

'''
def build_messages(input_text: str, few_shot_examples: List[Tuple[str, str]], k: int):
	"""
	Constructs messages for ASR correction where all user/assistant content is formatted as valid JSON strings.
	"""
	system_message = {
		"role": "system",
		"content": (
			"You are an expert ASR (Automatic Speech Recognition) generated transcript correction assistant. "
			"Your task is to check the 'asr_transcript' and fix any possible transcription issues like phonetic substitution errors, hallucinations, etc., while keeping the original meaning. "
			"Ignore punctuations, lower/upper case word issues, and don't change any words that seem correct. "
			"Generate the 'corrected_text' in a strictly valid JSON format."
		)
	}
	messages = [system_message]
	# Few-shot examples: Both User and Assistant content are valid JSON
	if k > 0 and few_shot_examples:
		for src, tgt in few_shot_examples:
			messages.append({
				"role": "user",
				"content": json.dumps({"asr_transcript": src})
			})
			messages.append({
				"role": "assistant",
				"content": json.dumps({"corrected_text": tgt})
			})
	# The actual task input: User content is valid JSON
	messages.append({
		"role": "user",
		"content": json.dumps({"asr_transcript": input_text})
	})
	# The Continuation Prompt: Starts the JSON structure for the model
	# Note: This is a partial JSON string to be completed by the model
	messages.append({
		"role": "assistant",
		"content": '{"corrected_text": "'
	})
	return messages
'''

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


# Demo Code Input:
#python vllmASR.py --model_type mistral --train_src data/asr_corrections/train.asr.txt --train_tgt data/asr_corrections/train.corrected.txt --test_src data/asr_corrections/test.asr.txt --test_tgt data/asr_corrections/test.corrected.txt --output_path outputs/mistral_asr_corrections.txt --k 3

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_type", type=str, choices=['mistral', 'llama', 'qwen', 'deepseek'], required=True)
	parser.add_argument("--train_src", type=str, required=True)
	parser.add_argument("--train_tgt", type=str, required=True)
	parser.add_argument("--test_src", type=str, required=True)
	parser.add_argument("--test_tgt", type=str, default=None)
	parser.add_argument("--output_path", type=str, default="")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--k", type=int, default=0)
	args = parser.parse_args()
	# Model Mapping
	model_map = {
		"mistral": "/model-weights/Mistral-7B-Instruct-v0.3/",
		"llama": "/model-weights/Meta-Llama-3.1-8B-Instruct/",
		"qwen": "/model-weights/Qwen2.5-7B-Instruct/",
		"deepseek": "/model-weights/DeepSeek-R1-Distill-Qwen-7B/"
	}
	model_id = model_map[args.model_type]
	# Load Data
	test_source = load_file(args.test_src)
	bm25 = None
	if args.k > 0:
		train_src = load_file(args.train_src)
		train_tgt = load_file(args.train_tgt)
		tok_train_src = [line.split() for line in train_src]
		tok_test_src = [line.split() for line in test_source]
		bm25 = bm25s.BM25(corpus=train_src)
		bm25.index(tok_train_src)
		resultsIDX, _ = bm25.retrieve(tok_test_src, k=args.k, corpus = range(len(train_src)))
		assert(len(resultsIDX) == len(test_source))
		print("BM25 indexing completed.")
		#bm25 = BM25Okapi([s.split() for s in train_src])
	# Initialize vLLM
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	llm = LLM(
		model=model_id,
		enable_prefix_caching = True,
		tensor_parallel_size = 1,
		max_model_len=2048,
		gpu_memory_utilization = 0.95
		)
	# We stop at the closing quote of the JSON field or the brace
	sampling_params = SamplingParams(
		temperature=0.1,
		#top_p=0.95,
		top_k=10,
		max_tokens=512,
		seed = args.seed,
		stop=["}"]
	)
	prompts = []
	#Use BM25S
	for iT in tqdm(range(len(test_source))):
		text = test_source[iT]
		few_shot = []
		if args.k > 0 and bm25:
			top_indices = resultsIDX[iT].tolist()
			few_shot = [(train_src[i], train_tgt[i]) for i in top_indices][::-1]
		messages = build_messages_joint(text, few_shot, args.k, args.model_type)
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
		if i == 0:
			print("Sample Generation:")
			print("ASR Transcript: ", test_source[i])
			print("Corrected Text: ", correction_only)
		final_output.append(correction_only)
		normalized_output.append(normalize_text(correction_only))
	# Save
	op = {}
	op["outputs"] = final_output
	op["normalized_outputs"] = normalized_output
	pkl.dump(op, open(args.output_path+".pkl", "wb"))
	x = open(args.output_path+"_norm.txt", "w", encoding="utf-8")
	for line in normalized_output:
		tmp = x.write(line+"\n")
	x.close()


if __name__ == "__main__":
	main()
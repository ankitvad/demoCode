import json
import argparse
import difflib
from typing import List, Tuple, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def get_edits(source: str, hypothesis: str, window_size: int = 2):
	"""Extracts edits with surrounding context for better LLM judgment."""
	s_tokens = source.split()
	h_tokens = hypothesis.split()
	matcher = difflib.SequenceMatcher(None, s_tokens, h_tokens)
	edits = []
	for tag, i1, i2, j1, j2 in matcher.get_opcodes():
		if tag == 'equal':
			continue
		
		# Define context window boundaries
		context_start = max(0, i1 - window_size)
		context_end = min(len(s_tokens), i2 + window_size)
		
		# Create a snippet showing context (e.g., "word1 word2 [EDIT_AREA] word3 word4")
		before_ctx = " ".join(s_tokens[context_start:i1])
		after_ctx = " ".join(s_tokens[i2:context_end])
		
		edit_info = {
			"type": tag,
			"from": " ".join(s_tokens[i1:i2]) if tag != 'insert' else "[EMPTY]",
			"to": " ".join(h_tokens[j1:j2]) if tag != 'delete' else "[REMOVED]",
			"context_snippet": f"{before_ctx} [[ {tag.upper()} ]] {after_ctx}",
			"full_source": source,
			"indices": (i1, i2)
		}
		edits.append(edit_info)
	return edits

def build_prompt(edit, tokenizer):
	"""Formats the edit with local context into a classification prompt."""
	
	# Specific instruction based on edit type
	if edit['type'] == 'insert':
		action_desc = f"inserting '{edit['to']}'"
	elif edit['type'] == 'delete':
		action_desc = f"deleting '{edit['from']}'"
	else:
		action_desc = f"changing '{edit['from']}' to '{edit['to']}'"
	instruction = (
		"You are an Automatic Speech Recognition (ASR) transcript checker. Review the proposed edit along with its local context. "
		"Does the proposed edit improve a possible transcription error or issue? "
		"Answer true/false for 'improves_transcript' in a strictly valid JSON format."
	)
	user_content = json.dumps({"Full Transcript": edit['full_source'], "Local Edit Context": edit['context_snippet'], "Proposed Edit": action_desc}, indent=4)
	messages = [
		{"role": "system", "content": instruction},
		{"role": "user", "content": user_content},
		{"role": "assistant", "content": '\n{"improves_transcript": "'}
	]
	#return messages
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)


def apply_edits(source, verified_edits):
	"""Reconstructs the sentence using only approved edits."""
	tokens = source.split()
	# Sort edits in reverse order to maintain index integrity during replacement
	for edit in sorted(verified_edits, key=lambda x: x['indices'][0], reverse=True):
		start, end = edit['indices']
		tokens[start:end] = edit['to'].split()
	return " ".join(tokens)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, required=True)
	parser.add_argument("--hypothesis", type=str, required=True)
	parser.add_argument("--model_type", type=str, choices=['mistral', 'llama', 'qwen', 'deepseek'], required=True)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	#model map:
	model_map = {
		"mistral": "/model-weights/Mistral-7B-Instruct-v0.3/",
		"llama": "/model-weights/Meta-Llama-3.1-8B-Instruct/",
		"qwen": "/model-weights/Qwen2.5-7B-Instruct/",
		"deepseek": "/model-weights/DeepSeek-R1-Distill-Qwen-7B/"
	}
	# Load Data
	sources = open(args.source, "r").read().strip().split("\n")
	hyps = open(args.hypothesis, "r").read().strip().split("\n")
	model_id = model_map[args.model_type]
	# Initialize vLLM
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	llm = LLM(
		model=model_id,
		enable_prefix_caching = True,
		tensor_parallel_size = 1,
		max_model_len=2048,
		gpu_memory_utilization = 0.95
		)
	sampling_params = SamplingParams(
		temperature=0.0,
		#top_p=0.95,
		#top_k=10,
		max_tokens=10,
		seed = args.seed,
		stop=["}", '"']
	)
	# Stage 1: Extract Edits and Build Prompts
	all_edits_per_sentence = []
	flat_prompts = []
	print("Extracting edits and building batch...")
	for s, h in zip(sources, hyps):
		sentence_edits = get_edits(s, h)
		all_edits_per_sentence.append(sentence_edits)
		for edit in sentence_edits:
			flat_prompts.append(build_prompt(edit, tokenizer))
	# Batch Inference
	if not flat_prompts:
		print("No edits found between files.")
		return
	outputs = llm.generate(flat_prompts, sampling_params)
	# Map predictions back to boolean
	predictions = []
	for out in outputs:
		# Expected text: "true" or "false"
		pred_text = out.outputs[0].text.strip().lower()
		assert(pred_text in ["true","false"])
		predictions.append("true" in pred_text)
	# Stage 2: Reconstruct Sentences
	pred_idx = 0
	final_results = []
	for i, s in enumerate(sources):
		current_sentence_edits = all_edits_per_sentence[i]
		approved_edits = []
		for edit in current_sentence_edits:
			if predictions[pred_idx]:
				approved_edits.append(edit)
			pred_idx += 1
		new_hyp = apply_edits(s, approved_edits)
		final_results.append(new_hyp)
	# Save Output
	with open(args.hypothesis.replace(".txt","-filtered.txt"), "w") as f:
		for line in final_results:
			f.write(line + "\n")
	print(f"Processed {len(sources)} sentences. Pseudo-hypothesis saved.")

if __name__ == "__main__":
	main()
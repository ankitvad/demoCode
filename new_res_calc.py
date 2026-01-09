import jiwer
from tqdm import tqdm

def get_word_operations(ref, hyp):
	"""Aligns strings and returns a dictionary of non-equal operations."""
	# Ensure inputs are strings
	ref, hyp = str(ref), str(hyp)
	out = jiwer.process_words(ref, hyp)
	ops = {}
	ref_tokens = ref.split()
	hyp_tokens = hyp.split()
	for op in out.alignments[0]:
		if op.type != "equal":
			ops[op.ref_start_idx] = {
				"type": op.type,
				"ref_content": " ".join(ref_tokens[op.ref_start_idx:op.ref_end_idx]),
				"hyp_content": " ".join(hyp_tokens[op.hyp_start_idx:op.hyp_end_idx]),
				"ref_indices": list(range(op.ref_start_idx, op.ref_end_idx))
			}
	return ops

def calculate_corpus_metrics(sources, targets, predictions):
	assert len(sources) == len(targets) == len(predictions), "List lengths must match."
	total_ref_words = 0
	total_words_fixed = 0
	total_words_introduced = 0
	total_old_errors = 0
	for s, t, p in tqdm(zip(sources, targets, predictions), total=len(sources), desc="Evaluating"):
		t_tokens = t.split()
		total_ref_words += len(t_tokens)
		# 1. Baseline: Count errors in the original ASR (Source vs Target)
		# Using jiwer.process_words to get the raw edit distance
		loc_error = jiwer.process_words(t, s)
		total_old_errors += loc_error.substitutions
		total_old_errors += loc_error.insertions
		total_old_errors += loc_error.deletions
		# 2. Alignment Logic
		required_ops = get_word_operations(s, t)
		predicted_ops = get_word_operations(s, p)
		# 3. Word-Level Attribution
		for idx, p_op in predicted_ops.items():
			if idx in required_ops:
				r_op = required_ops[idx]
				# CONTENT CHECK: Precise match of the fix
				if p_op['hyp_content'].strip().lower() == r_op['hyp_content'].strip().lower():
					total_words_fixed += len(r_op['ref_indices'])
				else:
					# Model attempted a fix but the content is wrong (Sub/Ins error)
					total_words_introduced += len(p_op['hyp_content'].split())
			else:
				# Model changed a correct word (Hallucination/Deletion/Insertion)
				total_words_introduced += len(p_op['hyp_content'].split())
				
	# 4. Final Metric Aggregation
	if total_ref_words == 0:
		return "Empty corpus."
	old_wer = total_old_errors / total_ref_words
	wer_corr = total_words_fixed / total_ref_words
	wer_intr = total_words_introduced / total_ref_words
	new_wer = old_wer - wer_corr + wer_intr
	# Calculate the percentage of existing errors that were successfully resolved
	correction_efficiency = (total_words_fixed / total_old_errors) * 100 if total_old_errors > 0 else 100
	return {
		"Corpus Stats": {
			"Total Ref Words": total_ref_words,
			"Total Baseline Errors": total_old_errors,
			"JIWER WER S_T": jiwer.wer(targets,sources),
			"JIWER WER P_T": jiwer.wer(targets,predictions)
		},
		"Refinement Metrics": {
			"Old WER": f"{old_wer:.4f}",
			"Correction WER (fixed)": f"{wer_corr:.4f}",
			"Introduction WER (added)": f"{wer_intr:.4f}",
			"New WER": f"{new_wer:.4f}"
		},
		"Efficiency": {
			"Correction Efficiency": f"{correction_efficiency:.2f}%",
			"Error Reduction": f"{(old_wer - new_wer) / old_wer * 100:.2f}%" if old_wer > 0 else "0%"
		}
	}

# --- Example Usage ---
source_list = ["the cat sst on mat", "he go to school"]
target_list = ["the cat sat on the mat", "he goes to school"]
pred_list   = ["the cat sat on mat", "he go to school"] # Fixed 1, missed 1

results = calculate_corpus_metrics(source_list, target_list, pred_list)

import json
print(json.dumps(results, indent=4))
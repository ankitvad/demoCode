import argparse
from jiwer import process_words
from typing import List, Dict


def read_file(path: str) -> List[str]:
	with open(path, "r", encoding="utf-8") as f:
		return [line.strip() for line in f if line.strip()]


def wer_edit_breakdown(
	references: List[str],
	hypotheses: List[str]
) -> Dict[str, float]:
	assert len(references) == len(hypotheses), (
		f"Mismatched lines: {len(references)} refs vs {len(hypotheses)} hyps"
	)
	result = process_words(references, hypotheses)
	S = result.substitutions
	D = result.deletions
	I = result.insertions
	N = result.reference_word_count
	wer = (S + D + I) / max(1, N)
	return {
		"wer": round(wer, 4),
		"insertions": I,
		"deletions": D,
		"substitutions": S,
		"num_ref_words": N,
		"ins_rate": round(I / max(1, N), 4),
		"del_rate": round(D / max(1, N), 4),
		"sub_rate": round(S / max(1, N), 4),
	}


def main():
	parser = argparse.ArgumentParser(
		description="Compute WER with insertion / deletion / substitution breakdown using JIWER"
	)
	parser.add_argument("--ref", required=True, help="Reference file (1 sentence per line)")
	parser.add_argument("--hypo", required=True, help="Hypothesis file (1 sentence per line)")
	args = parser.parse_args()
	refs = read_file(args.ref)
	hyps = read_file(args.hypo)
	metrics = wer_edit_breakdown(refs, hyps)
	print("\n===== WER Breakdown =====")
	for k, v in metrics.items():
		print(f"{k:15s}: {v}")
	print("========================\n")


if __name__ == "__main__":
	main()

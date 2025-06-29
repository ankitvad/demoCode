#parse all the trainer_state.json files and parse all the logging values and loss. Write it out in a txt file.
import glob
import json

log_files = glob.glob("*/*/checkpoint-1*/trainer_state.json")

writeOut = open("log_param.txt", "w")


def parse_results(log, fLog):
	train_loss = []
	eval_loss = []
	eval_wer_loss = []
	for entry in log['log_history']:
		if 'eval_loss' in entry:
			eval_loss.append(round(entry['eval_loss'], 4))
			eval_wer_loss.append(round(entry['eval_wer_loss'], 4))
		else:
			train_loss.append(round(entry['loss'], 4))
	#Print out the values in the writeOut txt file.
	tmp = writeOut.write("** " + fLog.replace("/trainer_state.json", "") + "\n")
	#Write out the whole list in a string format.
	tmp = writeOut.write("train_loss: " + str(train_loss) + "\n")
	tmp = writeOut.write("eval_loss: " + str(eval_loss) + "\n")
	tmp = writeOut.write("eval_wer_loss: " + str(eval_wer_loss) + "\n\n")


for fLog in log_files:
	with open(fLog, 'r') as f:
		log_history_data = json.load(f)
	parse_results(log_history_data, fLog)

writeOut.close()
print("Done parsing and writing out the log files.")


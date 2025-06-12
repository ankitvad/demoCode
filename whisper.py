#Speech to text model with Whisper-v3-turbo
#CUDA_VISIBLE_DEVICES=1 python whisper.py

import pickle as p
import glob
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from transformers.utils import logging
logging.set_verbosity_error()

modelLoc = "/datasets/model/whisper/"
opLoc = "/datasets/ankitUW/resources/stt_trnsc/whisper/"
#canary crisper whisper

en_audio = glob.glob("/datasets/resources/recordings/english/*.wav") + glob.glob("/datasets/resources/recordings/english/*/*.wav")

fr_audio = glob.glob("/datasets/resources/recordings/french/*.wav") +
glob.glob("/datasets/resources/recordings/french/*/*.wav")

import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = modelLoc

model = AutoModelForSpeechSeq2Seq.from_pretrained(
model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa")

model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
	"automatic-speech-recognition",
	model=model,
	tokenizer=processor.tokenizer,
	feature_extractor=processor.feature_extractor,
	chunk_length_s = 30,
	batch_size=64, #batch size for inference - set based on your device
	torch_dtype=torch_dtype,
	device =device,
	)

en_loc = {}
fr_loc = {}

cntr = 0

for e in tqdm(en_audio):
	result = pipe(e, generate_kwargs={"language": "english"}, return_timestamps=True)
	p.dump(result, open(opLoc+"EN/"+str(cntr)+".p", "wb"))
	en_loc[cntr] = e
	cntr += 1

p.dump(en_loc, open(opLoc+"en_loc.p", "wb"))


'''
for f in tqdm(fr_audio):
	result = pipe(f, generate_kwargs={"language": "french"}, return_timestamps=True)
	p.dump(result, open(opLoc+"FR/"+str(cntr)+".p", "wb"))
	fr_loc[cntr] = f
	cntr += 1

p.dump(fr_loc, open(opLoc+"fr_loc.p", "wb"))
'''

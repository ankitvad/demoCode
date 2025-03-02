'''
# prompt: install audio packages and transformer packages

!pip install transformers
!pip install datasets
!pip install librosa
!pip install soundfile
!pip install jiwer
!pip install pydub
!pip install datasets[audio]
'''

hf_model = "/datasets/model/"
sv_loc = ""

import os
import torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import pydub as pdb
import soundfile as sf
from datasets import load_dataset
import librosa
import torch
import numpy as np
from tqdm import tqdm
import pickle as p

#ASR Models
asr_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr", cache_dir = "hf_model/")
asr_model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr", cache_dir = "hf_model/")
'''
#TTS Models
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cache_dir = "hf_model/")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cache_dir = "hf_model/")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)
# Load Vocoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir = "hf_model/")
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model.to(device)

def displayTime(startFrame, endFrame, sr):
    print(' start time: ' + str(startFrame/sr) + ', end time: ' + str(endFrame/sr))

# Function to check whether the audio signal is mono channel or stereo and what the kHz/bitrate is.
# If the channel is stereo -> Convert to mono and if the kHz does not match - change to 16kHz.
def check_audio(audio_path):
    newSR = 16000
    time_interval_s = 30.00
    audio_data, sr = librosa.load(audio_path, sr=newSR, mono=True, dtype=np.float64)#sr=None) Downsample to 16kHz
    nonMuteSections = librosa.effects.split(audio_data, top_db=30)
    chunks = []
    begin = 0
    cntr = 0
    L = len(nonMuteSections)
    while cntr < L:
        if cntr != 0:
            if (nonMuteSections[cntr][1]/sr) - (nonMuteSections[begin][0]/sr) > time_interval_s:
                chunks.append([begin, cntr - 1])
                begin = cntr
        cntr += 1
    chunks.append([begin, L - 1])
    assert(chunks[-1][1] == L-1)
    M = len(chunks)
    for i in range(1,M,1):
        assert(chunks[i][0] == chunks[i-1][1] + 1)
    return chunks, audio_data, sr, nonMuteSections



#Load all the list of files to generate the transcript for:
en_audio = ""

for e in tqdm(en_audio):
    chunks, audio_data, sr, nonMuteSections = check_audio(e)
    trnscrpt_split = []
    for c in chunks:
        inputs = asr_processor(audio=audio_data[nonMuteSections[c[0]][0]:nonMuteSections[c[1]][1]], sampling_rate=sr, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        predicted_ids = asr_model.generate(**inputs, max_length=1000)
        transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        trnscrpt_split.append(transcription)
    svName = e.split('/')[-1].split('.')[0] + '.pkl'
    p.dump(trnscrpt_split, open(sv_loc + svName, 'wb'))


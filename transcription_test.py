from itertools import groupby

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import noisereduce as nr

import numpy as np
import matplotlib.pyplot as plt

model_name = 'facebook/wav2vec2-large-960h-lv60-self'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()

audio_path = r"uploads\e40f96c3-fb52-4ad5-8278-d92c20c895e6\audio.wav"

data, sampling_rate = librosa.load(audio_path, sr=16000)

# use 85th percentile

reduced = nr.reduce_noise(data, sampling_rate)
splits = librosa.effects.split(reduced, ref=np.average(np.abs(reduced)), frame_length=4096)
dists = []

for i in range(0,len(splits)):
    dist = splits[i][0] - splits[i-1][1]
    dists.append(dist)


# dists.sort()
split_time = np.percentile(dists, 75)

# plt.hlines(split_time, 0, len(dists))
# plt.scatter(np.arange(len(dists)), (dists))
# plt.show()
# 0/0


# print(dists[225])
# print(split_time)
phrases = [[]]

for i in range(len(splits) - 1):
    if dists[i] < split_time or len(phrases[-1]) == 0:
        phrases[-1].append(splits[i])
        if dists[i] > split_time:
            print(dists[i])
    else:
        phrases.append([splits[i]])

phrase_starts = [p[0][0] for p in phrases]
# print([a[0] for a in phrases][:10])
phrase_ends = [p[-1][1] for p in phrases]

# for s,e in splits:
#     plt.axvline(x=s, color='blue', linestyle='--')
#     plt.axvline(x=e, color='black', linestyle='--')


for s in phrase_starts:
    plt.axvline(x=s, color='green', linestyle='--')
for e in phrase_ends:
    plt.axvline(x=e, color='red', linestyle='--')


plt.plot(reduced)
plt.show()
0/0
input_values = processor(data, sampling_rate=sampling_rate, return_tensors="pt").input_values.cuda()

logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print(transcription)

words = transcription.split()
predicted_ids = predicted_ids[0].tolist()
duration_sec = input_values.shape[1] / sampling_rate

# ids_w_time_0 = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids) if _id != processor.tokenizer.pad_token_id]

ids_w_time = []
word = []

for i, _id in enumerate(predicted_ids):
    if _id == processor.tokenizer.pad_token_id:
        continue
    elif _id == processor.tokenizer.word_delimiter_token_id:
        if word:
            ids_w_time.append(word)
            word = []
        continue

    word.append((i/len(predicted_ids) * duration_sec, _id))

if word:
    ids_w_time.append(word)
    word = []



# split_ids_w_time = [list(group) for k, group in groupby(ids_w_time_0, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id) if not k]
# print(split_ids_w_time, len(split_ids_w_time))
print(words)
print(ids_w_time, len(ids_w_time), list(map(len, ids_w_time)))
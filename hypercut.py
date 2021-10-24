import os
import numpy as np

from google.cloud import speech, storage
# import torch
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import librosa
import soundfile as sf
import noisereduce as nr
from multiprocessing import cpu_count, Pool

import networkx as nx

import tqdm
import json

# model_name = 'facebook/wav2vec2-large-960h-lv60-self'
# model_name = 'facebook/wav2vec2-base-960h'
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()

RATE = 16_000
sentencer = SentenceTransformer("all-mpnet-base-v2")

# class Word:
#     def __init__(self, word:str):
#         self.start = None
#         self.end = None

#         self.str = word
#         self.letters = []

# def transcribe(audio:np.ndarray):
#     # later will return time level data, for now, will return just words
#     input_values = processor(audio, sampling_rate=RATE, return_tensors="pt").input_values.cuda()

#     logits = model(input_values).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.decode(predicted_ids[0])
#     return transcription

# def hypercut(audio:np.ndarray):
#     reduced = nr.reduce_noise(audio, RATE)
#     splits = librosa.effects.split(reduced, ref=np.average(np.abs(reduced)), frame_length=4096)
#     dists = []

#     for i in range(0,len(splits)):
#         dist = splits[i][0] - splits[i-1][1]
#         dists.append(dist)

#     split_time = np.percentile(dists, 75)

#     audio_phrases = [[]]

#     whole_transcription = []

#     for i in range(len(splits) - 1):
#         if dists[i] < split_time or len(audio_phrases[-1]) <= 1:
#             audio_phrases[-1].append(splits[i])
#             if dists[i] > split_time:
#                 print(dists[i])
#         else:
#             audio_phrases.append([splits[i]])

#     print(audio.shape)
#     for start,*_,end in tqdm.tqdm(audio_phrases):
#         whole_transcription.append(transcribe(audio[int(start[0]):int(end[1])]))

#     print(whole_transcription)

bucket_name = "hypercut-audio"

#src: https://github.com/maxzuo/Chatbot_Integration_Manager/blob/master/utils.py
# def cosineSimilarity(w, c):
#     num = np.dot(w.T, c)

#     wMag = np.sqrt(np.sum(np.multiply(w, w), axis=0))
#     wMag = np.expand_dims(wMag, axis=-1)

#     cMag = np.sqrt(np.sum(np.multiply(c, c), axis=0))
#     cMag = np.expand_dims(cMag, axis=-1)

#     res = num / np.dot(wMag, cMag.T)
#     np.fill_diagonal(res, 1)
#     return res

#src: https://github.com/maxzuo/Chatbot_Integration_Manager/blob/master/utils.py
def rank(simMat:np.ndarray):
    graph = nx.from_numpy_array(simMat)
    return nx.pagerank(graph, max_iter=1_000_000, tol=1e-4)

def transcribe(path:str, offset:float=0.):
    client = speech.SpeechClient()
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    dest_name = f"{os.path.basename(os.path.dirname(path))}-{os.path.basename(path)}"
    blob = bucket.blob(dest_name)

    blob.upload_from_filename(path)

    # audio, sample_rate = sf.read(path)
    # audio = speech.RecognitionAudio(content=open(path, 'rb').read())
    audio = speech.RecognitionAudio(uri=f"gs://{bucket_name}/{dest_name}")

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    response = operation.result(timeout=60 * 5)

    # print(path, offset, type(response), type(response.results), dir(response.results), type(response.results[0]), dir(response.results[0]))

    phrases = []

    for result in response.results:
        words = []
        for word in result.alternatives[0].words:
            words.append({
                "start": word.start_time.seconds + word.start_time.microseconds / 1e6 + offset,
                "word": word.word,
                "end": word.end_time.seconds + word.end_time.microseconds / 1e6 + offset,
            })
        phrase = {"start":words[0]['start'], "end": words[-1]['end'], "words":words}
        # if phrases:
        #     space = {"start":phrases[-1]['end'], "end":phrase['start'], "words":["_space"]}
        #     spaces.append(space)
        phrases.append(phrase)

    return phrases

def short_transcribe(path:str, offset:float=0.):
    client = speech.SpeechClient()

    audio, sample_rate = sf.read(path)
    audio = speech.RecognitionAudio(content=open(path, 'rb').read())

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    response = client.recognize(config=config, audio=audio)

    # print(path, offset, type(response), type(response.results), dir(response.results), type(response.results[0]), dir(response.results[0]))

    phrases = []
    # spaces = []

    for result in response.results:
        words = []
        for word in result.alternatives[0].words:
            words.append({
                "start": word.start_time.seconds + word.start_time.microseconds / 1e6 + offset / RATE,
                "word": word.word,
                "end": word.end_time.seconds + word.end_time.microseconds / 1e6 + offset / RATE,
            })
        if len(words) > 0:
            phrase = {"start":words[0]['start'], "end": words[-1]['end'], "words":words, "transcript":result.alternatives[0].transcript}
            # if phrases:
                # space = {"start":phrases[-1]['end'], "end":phrase['start'], "words":["_space"]}
                # spaces.append(space)
            phrases.append(phrase)


    return phrases


def hypercut(addr:str):
    audio, sr = librosa.load(addr, sr=RATE)
    reduced = nr.reduce_noise(audio, RATE)
    splits = librosa.effects.split(reduced, ref=np.average(np.abs(reduced)), frame_length=4096)
    dists = []

    for i in range(0,len(splits)):
        dist = splits[i][0] - splits[i-1][1]
        dists.append(dist)

    split_time = np.percentile(dists, 75)

    audio_phrases = [[]]

    for i in range(len(splits) - 1):
        if dists[i] < split_time or len(audio_phrases[-1]) <= 2:
            audio_phrases[-1].append(splits[i])
        else:
            audio_phrases.append([splits[i]])
    audio_phrases[-1].append(splits[-1])

    audio_paths = []
    for i in range(len(audio_phrases)):
        p = os.path.join(os.path.dirname(addr), f"{i}.flac")
        audio_paths.append(p)
        sf.write(p, audio[int(audio_phrases[i][0][0]):int(audio_phrases[i][-1][1])], RATE, format="flac")
    # sf.write(os.path.join(os.path.dirname(addr), "audio.flac"), audio, RATE, format="flac")

    # full transcription:
    # transcriptions = transcribe(os.path.join(os.path.dirname(addr), "audio.flac"), 0)

    # split
    phrases = []
    spaces = []
    sentences = []

    with Pool(cpu_count()//2) as pool:
        for res in pool.starmap(short_transcribe, zip(audio_paths, [a[0][0] for a in audio_phrases])):
            phrases.extend(res)

    spaces.append({"start":0., "end":phrases[0]['start'], "words":[{"start":0., "end":phrases[0]['start'], "word":"_space"}]})
    for i in range(len(phrases)-1):
        spaces.append({"start":phrases[i]['end'], "end":phrases[i+1]['start'], "words":[{"start":phrases[i]['end'], "end":phrases[i+1]['start'], "word":"_space"}]})
        sentences.append(phrases[i]['transcript'])
    sentences.append(phrases[-1]['transcript'])

    embeddings = sentencer.encode(sentences)
    pca = PCA()
    flat_embeddings = pca.fit_transform(embeddings)

    cs = ((1 - cdist(flat_embeddings, flat_embeddings, 'cosine')+1)/2 * 0.9) + 0.1
    rankings = rank(cs)
    for ranking,index in enumerate(sorted(rankings, key=lambda a: rankings[a])):
        phrases[index]['priority'] = ranking + 1
    # print(transcriptions)

    # with open("test_priority.json", 'w') as f:
    #     json.dump(phrases, f, indent=2)


    return phrases, spaces

if __name__ == "__main__":
    # data, sr = librosa.load(r"uploads\5cd5f40f-d7bf-4b7f-a76c-054c06d03566\audio.wav", sr=RATE)
    import time

    start = time.time()
    hypercut(r"uploads\5cd5f40f-d7bf-4b7f-a76c-054c06d03566\audio.wav")
    print(time.time() - start)
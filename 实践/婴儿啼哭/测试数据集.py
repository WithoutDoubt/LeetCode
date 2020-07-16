import os
import wave
import librosa
import numpy as np
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt

DATA_DIR = './test'

file_glob = []


def track_features(y, sr):

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=33)
    feature_0 = mfcc.T

    return feature_0


def get_wave_norm(file):
    with wave.open(file, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        data = f.readframes(nframes)
    data = np.fromstring(data, dtype=np.int16)
    # data = data * 1.0 / max(abs(data))
    return data, framerate


seg = 250000

data_f = {}

for file in tqdm(os.listdir(DATA_DIR)):
    data_x = []
    raw, sr = get_wave_norm(os.path.join(DATA_DIR, file))
    length = raw.shape[0]
    for i in range((length//seg)*2+1):
        start = i * int(seg // 2)
        end = start + seg
        if end > length:
            end = length
        x = np.zeros(seg)
        x[start-end:] = raw[start:end]
        r = track_features(x, sr)
        data_x.append(r)
    data_f[file] = data_x


with open('./data_test.pkl', 'wb') as f:
    pkl.dump(data_f, f)

for key, value in data_f.items():
    print(key, len(value))
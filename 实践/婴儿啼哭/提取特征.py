import os
import wave
import librosa
import numpy as np 
from tqdm import tqdm
import pickle as pkl 
from sklearn.preprocessing import normalize

def extract_logmel (y, sr,size=3):
    """
    extract log mel spectrogram feature
    : param y: the input signal (audio time series)
    : param sr: sample rate of 'y'
    : param size: the length (seconds) of random crop from original audio, default as 3 seconds 
    """
    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    if len(y) <= size * sr:
        new_y = np.zeros((size * sr + 1,))
        new_y[:len(y)] = y
        y = new_y

    start = np.random.randint(0,len(y)-size*sr)  # 随机选取一个开始点
    y = y[start : start + size * sr]               # 随机截取一下 y

    melspectrogram = librosa.feature.melspectrogram(y = y,
                                                    sr = sr,
                                                    n_fft = 2048,
                                                    hop_length = 1024,
                                                    n_mels = 60)

    logmelspec = librosa.power_to_db(melspectrogram)

    return logmelspec

def get_wave_norm(file):
    data, framerate = librosa.load(file, sr = 22050)
    return data,framerate

LABELS = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable']
DATA_DIR = './train'
file_glob = []
data = []

for i , cls_fold in enumerate(os.listdir(DATA_DIR)):
    cls_base = os.path.join(DATA_DIR,cls_fold)
    lbl = cls_fold

    files = os.listdir(cls_base)
    print('{} train num:'.format(lbl),len(files))
    for pt in files:
        file_pt = os.path.join(cls_base,pt)
        file_glob.append((file_pt,LABELS.index(lbl))) 

print("done")
print(len(file_glob))

for fileone, lbl in tqdm(file_glob):
    try:
        raw,sr = get_wave_norm(fileone)
    except Exception as e:
        print(e,fileone)
    feature = extract_logmel(y = raw, sr = sr, size = 15)           # 15 s 是不对的
    y = np.zeros(len(LABELS))
    y[lbl] = 1
    data.append((feature,y))

with open('./data.pkl', 'wb') as f:
    pkl.dump(data,f)

del data



import os
import wave
import numpy as np 
import pickle as pkl 

train_x = []
train_y = []

LABELS = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable']
N_CLASS = len(LABELS)

with open ('./data.pkl','rb') as f:
    raw_data = pkl.load(f)

np.random.seed(5)
np.random.shuffle(raw_data)

print(raw_data[0][0].shape)

train_data = raw_data[:-50]
valid_data = raw_data[-50:]

print (len(train_data))
print (len(valid_data))
print (train_data[0][0].shape)

# 868
# 50
# (60, 323)

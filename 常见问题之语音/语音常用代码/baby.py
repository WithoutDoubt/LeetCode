import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

DATA_PATH = '../input/train/'

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    print(labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def wav2mfcc(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(wave, sr=8000)

    return mfcc


def save_data_to_array(path=DATA_PATH):
    labels, _, _ = get_labels(path)

    for label in labels:
        mfcc_vectors = []
        
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = np.zeros((20, 400))
            mfcc_feat = wav2mfcc(wavfile)[:, :400]
            mfcc[:, :mfcc_feat.shape[1]] = mfcc_feat
            mfcc_vectors.append(mfcc)
            
        mfcc_vectors = np.stack(mfcc_vectors)
            
        
        np.save(label + '.npy', mfcc_vectors)

        
DATA_TEST_PATH = '../input/test'
def save_data_to_array_test(path=DATA_TEST_PATH):
    mfcc_vectors = []
        
    wavfiles = [DATA_TEST_PATH + '/' + wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format('test')):
        mfcc = np.zeros((20, 400))
        mfcc_feat = wav2mfcc(wavfile)[:, :400]
        mfcc[:, :mfcc_feat.shape[1]] = mfcc_feat
        mfcc_vectors.append(mfcc)
            
    mfcc_vectors = np.stack(mfcc_vectors)
    np.save('test.npy', mfcc_vectors)
        

def get_train_test(split_ratio=0.8, random_state=42):
    labels, indices, _ = get_labels(DATA_PATH)


    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])


    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    return X, y

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(20, 400, channel)))
    # 32 个filter，输出为  (20 - 3)/ stride + 1 = 18

    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    # 输出为  (18 - 3)/ stride + 1 = 16

    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    # 输出为  (16 - 3)/ stride + 1 = 14

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 14/2 = 7

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


#################################################
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

feature_dim_2 = 32


# save_data_to_array()
# save_data_to_array_test()

X, Y = get_train_test()
skf = StratifiedKFold(n_splits=5)

test_pred = np.zeros((228, 6))
for idx, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):
    print(idx)

    feature_dim_1 = 20
    channel = 1
    epochs = 50
    batch_size = 6
    verbose = 1
    num_classes = 6

    X_train, X_test = X[tr_idx], X[val_idx]
    y_train, y_test = Y[tr_idx], Y[val_idx]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel) / 255.0
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel) / 255.0

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    
    model = get_model()

    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ModelCheckpoint(filepath='model-{0}.h5'.format(idx), save_best_only=True),
    ]

    model.fit(X_train, y_train_hot, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=verbose, 
              validation_data=(X_test, y_test_hot),
              callbacks=my_callbacks
             )
    model.load_weights('model-{0}.h5'.format(idx))
    
    X_test = np.load('test.npy') / 255.0
    test_pred += model.predict(X_test.reshape(228, 20, 400, 1))


test_pred = np.zeros((228, 6))
for path in ['model-0.h5', 'model-2.h5', 'model-6.h5'][:1]:
    model.load_weights(path)
    
    X_test = np.load('test.npy') / 255.0
    test_pred += model.predict(X_test.reshape(228, 20, 400, 1))


wavfiles = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]


###################################################################
import pandas as pd
df = pd.DataFrame()

df['id'] = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
df['label'] = [['hug', 'sleepy', 'uncomfortable', 'hungry', 'awake', 'diaper'][x] for x in test_pred.argmax(1)]
df.to_csv('tmp225.csv', index=None)

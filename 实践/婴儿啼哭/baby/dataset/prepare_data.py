import os
import random
import sys 
sys.path.append('/home/ubuntu/yww/IRMLSTM')
import librosa
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm
import hpyter as hp
from util.utils import synthesis_noisy_y
from util.metrics import compute_PESQ, compute_STOI

import pickle as pkl


class IRMDataset(Dataset):
    def __init__(self,
                 file_path="/home/ubuntu/yww/Script/data_mfcc_train.pkl",
                 mode="train",
                 n_jobs=-1
                 ):
        """Construct training dataset.

        Args:
            noise_dataset: List, which saved the paths of noise files.
            clean_dataset: List, which saved the paths of clean wav files.
            offset: offset of clean_dataset.
            limit: limit of clean_dataset from offset position.
            n_jobs: Use multithreading to pre-load noise files, see joblib (https://joblib.readthedocs.io/en/latest/parallel.html).
            mode:
                "train": return noisy magnitude, mask.
                "validation": return noisy_y, clean_y, name
                "test": return noisy_y, name

                
                ...
        """
        super().__init__()
        assert mode in ["train", "validation", "test"], "mode parameter must be one of 'train', 'validation', and 'test'."
       
        
        with open(file_path,'rb') as f:
            raw_data = pkl.load(f)

        
        self.length = len(raw_data)
        self.train_data = raw_data[:-50]
        self.valid_data = raw_data[-50:]
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # clean_y, _ = librosa.load(self.clean_f_paths[idx], sr=16000)
        # snr = random.choice(self.snr_list)

        # noise_data = random.choice(self.all_noise_data)
        all_data_train = random.choice(self.train_data)
        all_data_valid = self.valid_data
        # noise_name = noise_data["name"]
        # noise_y = noise_data["y"]

        # name = f"{str(idx).zfill(5)}_{noise_name}_{snr}"
        # clean_y, noise_y, noisy_y = synthesis_noisy_y(clean_y, noise_y, snr)

        x = all_data_train[idx][0]
        y = all_data_train[idx][1]

        if self.mode == "train":
        
            # clean_mag, _ = librosa.magphase(librosa.stft(clean_y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.n_fft))
            # noise_mag, _ = librosa.magphase(librosa.stft(noise_y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.n_fft))
            # noisy_mag, _ = librosa.magphase(librosa.stft(noisy_y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.n_fft))
            # mask = np.sqrt(clean_mag ** 2 / (clean_mag + noise_mag) ** 2)
            # n_frames = clean_mag.shape[-1]
            x = all_data_train[idx][0]
            y = all_data_train[idx][1]
            return x, y
        elif self.mode == "validation":
            x_valid = all_data_valid[idx][0]
            y_valid = all_data_valid[idx][1]
            return x_valid, y_valid 
        else:
            return noisy_y, name


if __name__ == '__main__':
    dataset = IRMDataset(mode='validation')
    res = next(iter(dataset))
    print(type(res))


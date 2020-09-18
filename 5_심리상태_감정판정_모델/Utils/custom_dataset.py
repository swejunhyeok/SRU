import torch
import os
import cv2
import time
import numpy as np

from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self, train = True):
        self.train = train
        if train:
            self.video = np.load('./Dataset/train_video.npy')
            self.audio = np.load('./Dataset/train_mfccs.npy')
            self.y = np.load('./Dataset/train_y.npy')
        else:
            self.file_dir = []
            f = open('./Dataset/test_list.txt', 'r')
            while True:
                line = f.readline()
                if not line: break
                self.file_dir.append(line.strip())
            self.video = np.load('./Dataset/test_video.npy')
            self.audio = np.load('./Dataset/test_mfccs.npy')
            self.y = np.load('./Dataset/test_y.npy')
        print("Dataset Load Complete!")
        print("===========",self.video.shape,self.audio.shape,self.y.shape,"==============")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        video = torch.FloatTensor(self.video[idx])
        video = video.permute(3, 2, 1, 0)
        audio = torch.FloatTensor(self.audio[idx])
        audio = torch.unsqueeze(audio, 0)
        y = int(self.y[idx])
        if self.train:
            return video, audio, y
        else:
            return video, audio, y, self.file_dir[idx]




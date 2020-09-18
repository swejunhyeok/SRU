import torch
import os
import numpy as np
import torch

from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self, video_file, label_file):
        self.label_file = label_file
        self.video = np.load(video_file)
        self.label = np.load(label_file)  

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        video = self.video[idx]
        video = torch.FloatTensor(video)
        video = video.permute(3, 2, 1, 0)

        if self.label_file == "./Dataset/train_hr.npy" or self.label_file == "./Dataset/test_hr.npy":
            label = []
            label.append(int(self.label[idx]) - 80)
            label = np.asarray(label)
            label = torch.FloatTensor(np.asarray(label))
        else:
            label = []
            label.append(int(self.label[idx]))
            label = np.asarray(label)
            label = torch.FloatTensor(np.asarray(label))
        return video, label
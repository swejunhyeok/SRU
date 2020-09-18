import torch
import os
import numpy as np
import torch

from torch.utils.data import Dataset

class test_custom_dataset(Dataset):
    def __init__(self, video_file):
        self.video = np.load(video_file)
        self.hr_label = []
        self.br_label = []
        self.dir_file = []
        f=open('./Dataset/test_list.txt', 'r')
        while True:
                line = f.readline()
                if not line: break
                self.dir_file.append(line.split('\t')[0])
        f=open('./Dataset/test_label.txt')
        while True:
                line = f.readline()
                if not line: break
                self.hr_label.append(int(line.split('\t')[0]))
                self.br_label.append(int(line.split('\t')[1].strip()))

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        video = self.video[idx]
        video = torch.FloatTensor(video)
        video = video.permute(3, 2, 1, 0)

        hr_label = []
        hr_label.append(int(self.hr_label[idx]) - 80)
        hr_label = np.asarray(hr_label)
        hr_label = torch.FloatTensor(np.asarray(hr_label))
        
        br_label = []
        br_label.append(int(self.br_label[idx]))
        br_label = np.asarray(br_label)
        br_label = torch.FloatTensor(np.asarray(br_label))
        return video, hr_label, br_label, self.dir_file[idx]
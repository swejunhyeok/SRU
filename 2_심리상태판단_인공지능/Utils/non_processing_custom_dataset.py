import torch
import librosa
import os
import cv2
import time
import numpy as np
import warnings
warnings.simplefilter("ignore")
from torch.utils.data import Dataset

class non_processing_custom_dataset(Dataset):
    def __init__(self, Dataset_root, file_list):
        lst = []
        count = 0
        f = open(file_list, 'r')
        self.file_dir = []
        while True:
            line = f.readline()
            if not line: break
            count += 1
            if count % 100 == 0 :
                print(count)
            file_name = Dataset_root + line.strip()
            self.file_dir.append(line.strip())
            
            X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T,axis=0)

            capture = cv2.VideoCapture(file_name)

            file = int(line.strip()[7:8]) - 1

            frame_num = 16
            total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1
            df = (int)(total_frame / frame_num)

            video = []
            while(capture.isOpened()):
                if capture.get(cv2.CAP_PROP_POS_FRAMES) + 1 == capture.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
                elif (capture.get(cv2.CAP_PROP_POS_FRAMES) + 1) % df != 0 :
                    ret, frame = capture.read()
                    continue
                ret, frame = capture.read()
                ''' face detection '''
                face_cascade = cv2.CascadeClassifier('./Utils/haarcascade_frontface.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) != 0:
                    x,y, w, h = faces[0]
                    frame = frame[y:y+h, x:x+w]
                frame = cv2.resize(frame, (128, 128))
                video.append(frame)
                if len(video) == frame_num:
                    break

            arr = video, mfccs, file
            lst.append(arr)

            video, mfccs, y = zip(*lst)

            self.video = np.asarray(video)
            self.audio = np.asarray(mfccs)
            self.y = np.asarray(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        video = torch.FloatTensor(self.video[idx])
        video = video.permute(3, 2, 1, 0)
        audio = torch.FloatTensor(self.audio[idx])
        audio = torch.unsqueeze(audio, 0)
        y = int(self.y[idx])
        return video, audio, y, self.file_dir[idx]
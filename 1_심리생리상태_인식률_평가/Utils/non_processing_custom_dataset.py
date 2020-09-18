import torch
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
        self.dir_file = []
        while True:
            line = f.readline()
            if not line: break
            count += 1
            if count % 100 == 0 :
                print(count)
            file_name = Dataset_root + line.split('\t')[0].strip()
            self.dir_file.append(line.split('\t')[0].strip())

            capture = cv2.VideoCapture(file_name)
            
            hr_label = line.split('\t')[1].strip()
            br_label = line.split('\t')[2].strip()

            frame_num = 128
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

            arr = video, hr_label, br_label
            lst.append(arr)

            video, hr_label, br_label= zip(*lst)

            self.video = np.asarray(video)
            self.hr_label = np.asarray(hr_label)
            self.br_label = np.asarray(br_label)

    def __len__(self):
        return len(self.hr_label)

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
import argparse
import os
import csv

import torch
import time
from models.physnet import physnet
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from Utils.test_custom_dataset import test_custom_dataset

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 32)')

def main():
    global args
    args = parser.parse_args()

    test_dataset = test_custom_dataset("./Dataset/test_video.npy")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    hr_model = physnet()
    br_model = physnet()

    hr_model = hr_model.cuda()
    br_model = br_model.cuda()

    hr_checkpoint = torch.load('./check_point/hr.pth.tar')
    hr_model.load_state_dict(hr_checkpoint['state_dict'])

    br_checkpoint = torch.load('./check_point/br.pth.tar')
    br_model.load_state_dict(br_checkpoint['state_dict'])

    criterion = nn.L1Loss().cuda()

    hr_model.eval()
    br_model.eval()
    test_loss = test(test_loader, hr_model, br_model, criterion)

def test(test_loader, hr_model, br_model, criterion):
    hr_total_loss = 0.0
    br_total_loss = 0.0
    total_csv_logger = CSVLogger(args=args, fieldnames=['video_name', 'HR_label', 'HR_predict', 'HR_MAE/BPM', 'BR_label', 'BR_predict', 'BR_MAE/BPM'], filename='Total_Result.csv')
    for i, (video_appear, hr_target, br_target, dir) in enumerate(test_loader):
        test_csv_logger = CSVLogger(args=args, fieldnames=['video_name', 'HR_label', 'HR_predict', 'HR_MAE/BPM', 'BR_label', 'BR_predict', 'BR_MAE/BPM'], filename=str(i+1) + '_Test_Result.csv')
        video_appear = video_appear.cuda()
        hr_target = hr_target.cuda()
        br_target = br_target.cuda()

        with torch.no_grad():
            hr_output = hr_model(video_appear)
            br_output = br_model(video_appear)

        hr_loss = criterion(hr_output, hr_target)
        br_loss = criterion(br_output, br_target)

        for idx in range(hr_target.size(0)):
            row = {'video_name': str(dir[idx]), 'HR_label': str(hr_target.data[idx].item()+80), 'HR_predict': str(hr_output[idx].item()+80), 'HR_MAE/BPM': str(abs(hr_target.data[idx].item() - hr_output[idx].item())), 'BR_label': str(br_target.data[idx].item()), 'BR_predict': str(br_output[idx].item()), 'BR_MAE/BPM': str(abs(br_target.data[idx].item() - br_output[idx].item()))}
            total_csv_logger.writerow(row)
            test_csv_logger.writerow(row)
        row = {'video_name': '-', 'HR_label': '-', 'HR_predict': '-', 'HR_MAE/BPM': str(hr_loss.item()), 'BR_label': '-', 'BR_predict': '-', 'BR_MAE/BPM': str(br_loss.item())}
        test_csv_logger.writerow(row)
        test_csv_logger.close()
        hr_total_loss += hr_loss.item()
        br_total_loss += br_loss.item()
    hr_total_loss /= (i + 1)
    br_total_loss /= (i + 1)
    row = {'video_name': '-', 'HR_label': '-', 'HR_predict': '-', 'HR_MAE/BPM': str(hr_total_loss), 'BR_label': '-', 'BR_predict': '-', 'BR_MAE/BPM': str(br_total_loss)}
    total_csv_logger.writerow(row)
    total_csv_logger.close()

class CSVLogger():
    def __init__(self, args, fieldnames, filename='result.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

if __name__ == '__main__':
    main()


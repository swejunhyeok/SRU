import argparse
import os
import csv

import torch
import time
from models.fusion_net import fusion_net
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from Utils.custom_dataset import custom_dataset

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training (default: 32)')

def main():
    global args

    args = parser.parse_args()

    test_dataset = custom_dataset(False)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = fusion_net()
    model = model.cuda()

    checkpoint = torch.load('./check_point/model.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()

    total_csv_logger = CSVLogger(args=args, fieldnames=['video_name', 'video_label', 'predict_label', 'corret'], filename='Total_Result.csv')

    model.eval()
    test_acc, test_loss = test(test_loader, model, criterion, total_csv_logger)

    total_csv_logger.close()

def test(loader, model, criterion, total_csv_logger):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    label_word = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    for i, (video, audio, target, dir) in enumerate(loader):
        test_csv_logger = CSVLogger(args=args, fieldnames=['video_name', 'video_label', 'predict_label', 'corret'], filename=str(i+1)+'_test_Result.csv')
        # measure data loading time
        video = video.cuda()
        audio = audio.cuda()
        target = target.type(dtype=torch.long)
        target = target.cuda()
        
        with torch.no_grad():
            output = model(audio, video)

        loss = criterion(output, target)
        output = torch.max(output.data, 1)[1]
        total += target.size(0)

        cor = (output == target.data).sum().item()
        correct += (output == target.data).sum().item()
        xentropy_loss_avg += loss.item()

        for idx in range(target.size(0)):
            row = {'video_name': str(dir[idx]), 'video_label': label_word[target.data[idx]], 'predict_label': str(label_word[output[idx]]), 'corret': str((output[idx] == target.data[idx]).item())}
            total_csv_logger.writerow(row)
            test_csv_logger.writerow(row)
        row = {'video_name': '-', 'video_label': '-', 'predict_label': 'accuracy = ', 'corret': str(cor/target.size(0) * 100)}
        test_csv_logger.writerow(row)
        test_csv_logger.close()

    accuracy = correct / total
    test_loss = xentropy_loss_avg / i
    row = {'video_name': '-', 'video_label': '-', 'predict_label': 'accuracy = ', 'corret': str(accuracy * 100)}
    total_csv_logger.writerow(row)
    return accuracy, test_loss

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

import argparse
import os

import torch
import time
from models.physnet import physnet
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader
from Utils.custom_dataset import custom_dataset

dataset_options = ['hr', 'br']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='hr',
                    choices=dataset_options)
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--lr', type=float, default=1,
                    help='learning rate')
parser.add_argument('--rho', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--eps', type=float, default=1e-8,
                     help='learning decay for lr scheduler')

def main():
    global args
    args = parser.parse_args()

    if args.dataset == 'hr':
        train_dataset = custom_dataset("./Dataset/train_video.npy", "./Dataset/train_hr.npy")
        test_dataset = custom_dataset("./Dataset/test_video.npy", "./Dataset/test_hr.npy")
    elif args.dataset == 'br':
        train_dataset = custom_dataset("./Dataset/train_video.npy", "./Dataset/train_br.npy")
        test_dataset = custom_dataset("./Dataset/test_video.npy", "./Dataset/test_br.npy")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = physnet()

    model = model.cuda()

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    
    optimizer = optim.Adadelta(filtered_parameters, lr=args.lr, rho=args.rho, eps=args.eps)

    criterion = nn.L1Loss().cuda()

    best_loss = 100000
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader)
        model.train()
        train_loss = train(progress_bar, model, criterion, optimizer, epoch)
        model.eval()
        test_loss = test(test_loader, model, criterion, epoch)
        tqdm.write('train_loss: {0:.3f} / test_loss: {1:.3f}'.format(train_loss, test_loss))

        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint_name = args.dataset
            save_checkpoint({
                'epoch': epoch,
                'dataset' : args.dataset,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, checkpoint_name)
            tqdm.write('save_checkpoint')

def train(progress_bar, model, criterion, optimizer, epoch):
    total_loss = 0.0
    for i, (video_appear, target) in enumerate(progress_bar):
        video_appear = video_appear.cuda()
        target = target.cuda()

        output = model(video_appear)

        loss = criterion(output, target)

        total_loss += loss.item()

        progress_bar.set_description('Epoch : {0} / loss :{1:.3f}'.format(epoch, loss.item()))
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss /= (i + 1)
    return total_loss

def test(test_loader, model, criterion, epoch):
    total_loss = 0.0
    for i, (video_appear, target) in enumerate(test_loader):
        video_appear = video_appear.cuda()
        target = target.cuda()

        with torch.no_grad():
            output = model(video_appear)

        loss = criterion(output, target)

        total_loss += loss.item()
    total_loss /= (i + 1)
    return total_loss

def save_checkpoint(state, test_id):
    if not(os.path.isdir('check_point')):
        os.makedirs(os.path.join('check_point'))
    filename = 'check_point/' + test_id +'.pth.tar'
    torch.save(state, filename)

if __name__ == '__main__':
    main()


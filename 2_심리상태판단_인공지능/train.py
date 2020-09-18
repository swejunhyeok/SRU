import argparse
import os

from tqdm import tqdm
import torch
import time
from models.fusion_net import fusion_net
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

from Utils.custom_dataset import custom_dataset

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--lr', type=float, default=1,
                    help='learning rate')
parser.add_argument('--rho', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--eps', type=float, default=1e-8,
                     help='learning decay for lr scheduler')
parser.add_argument('--momentum', type=float, default=0.9,  metavar='M',
                    help='momentum')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--lr_decay', type=float, default=1e-4,
                     help='learning decay for lr scheduler')

def main():
    global args

    args = parser.parse_args()

    model_path = 'check_point'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_dataset = custom_dataset(True)
    test_dataset = custom_dataset(False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = fusion_net()

    model = model.cuda()

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    #print('Trainable params num : ', sum(params_num))
    
    optimizer = optim.Adadelta(filtered_parameters, lr=args.lr, rho=args.rho, eps=args.eps)

    criterion = nn.CrossEntropyLoss().cuda()

    best_acc = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader)

        # train for one epoch4
        model.train()
        train_acc, train_loss = train(progress_bar, model, criterion, optimizer, epoch)

        model.eval()
        test_acc, test_loss = test(test_loader, model, criterion)

        tqdm.write('train_loss: {0:.3f} train_acc: {1:.3f} / test_loss: {2:.3f} test_acc: {3:.3f}'.format(train_loss, train_acc, test_loss, test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch,4
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, model_path, 'model')
            tqdm.write('save_checkpoint')
    csv_logger.close()


def train(progress_bar, model, criterion, optimizer, epoch):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    losses = 0

    for i, (video, audio, target) in enumerate(progress_bar):
        video = video.cuda()
        audio = audio.cuda()
        target = target.type(dtype=torch.long)
        target = target.cuda()

        output = model(audio, video)

        loss = criterion(output, target)

        progress_bar.set_description('Epoch : {0} / loss :{1:.3f}'.format(epoch, loss.item()))
        output = torch.max(output.data, 1)[1]

        total += target.size(0)
        correct += (output == target.data).sum().item()
        xentropy_loss_avg += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate running average of accuracy
    accuracy = correct / total
    train_loss = xentropy_loss_avg / i

    progress_bar.set_postfix(xentropy='%.3f' % (train_loss), acc='%.3f' % accuracy)

    return accuracy, train_loss

def test(loader, model, criterion):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    for i, (video, audio, target) in enumerate(loader):
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

        correct += (output == target.data).sum().item()
        xentropy_loss_avg += loss.item()

    accuracy = correct / total
    test_loss = xentropy_loss_avg / i

    return accuracy, test_loss

def save_checkpoint(state, model_path, test_id):
    filename = model_path + '/' + test_id +'.pth.tar'
    torch.save(state, filename)

if __name__ == '__main__':
    main()

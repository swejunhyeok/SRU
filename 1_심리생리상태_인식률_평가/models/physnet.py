import torch.nn as nn
import torch
import numpy

class physnet(nn.Module):
    def __init__(self):
        super(physnet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])

        self.conv2 = nn.Conv3d(64, 128, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
        self.bn2 = nn.BatchNorm3d(128)
        
        self.conv3 = nn.Conv3d(128, 256, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, 1024, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
        self.bn5 = nn.BatchNorm3d(1024)

        self.conv6 = nn.Conv3d(1024, 2048, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
        self.bn6 = nn.BatchNorm3d(2048)

        self.conv7 = nn.Conv3d(2048, 2048, [3, 3, 3], [2, 2, 2], padding=[1, 1, 1])
        self.bn7 = nn.BatchNorm3d(2048)

        self.fc1 = nn.Linear(16384, 2048) 

        self.fc2 = nn.Linear(2048, 512) 

        self.fc3 = nn.Linear(512, 32) 

        self.fc4 = nn.Linear(32, 1)

    def forward(self, video):
        video = self.conv1(video)

        video = self.conv2(video)
        video = self.bn2(video)

        video = self.conv3(video)
        video = self.bn3(video)

        video = self.conv4(video)
        video = self.bn4(video)

        video = self.conv5(video)
        video = self.bn5(video)

        video = self.conv6(video)
        video = self.bn6(video)

        video = video.view(video.size(0), -1)

        video = self.fc1(video)

        video = self.fc2(video)

        video = self.fc3(video)

        output = self.fc4(video)

        return output




    

      








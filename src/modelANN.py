import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import dropout
from typing import Tuple
import torch


class MortalityANN(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int] = (3, 128, 128), first_layer_conv2d: bool = False):
        super(MortalityANN, self).__init__()

        self.downsample_factor: int = 8  # Factor of 8 bc we use 3 layers of max pooling. I.e. 2**3 = 8
        self.input_size = input_size
        self.first_layer_conv2d = first_layer_conv2d

        assert self.input_size[1] % self.downsample_factor == 0
        assert self.input_size[2] % self.downsample_factor == 0

        # cnn(in_channel, out_channel, kernel_size, stride, padding)
        # self.conv1 = nn.Conv1d(1, 32, kernel_size=9, padding=4)
        if self.first_layer_conv2d:
            self.conv1 = nn.Conv2d(input_size[0], 64 * 9, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64  * 9)
        else:
            self.conv1 = nn.Linear(np.prod(self.input_size), 64)
            self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Linear(64 * 2 * 9, 64 * 9)  # multiplying by 9 to match CNN # of parameters
        self.bn2 = nn.BatchNorm1d(64 * 9)
        self.conv3 = nn.Linear(64 * 9, 128 * 9)  # , kernel_size= 3, padding= 1)
        self.bn3 = nn.BatchNorm1d(128 * 9)
        self.fc1 = nn.Linear(128 * 9, 120)  # * int(self.input_size[1] / self.downsample_factor) *  int(self.input_size[2] / self.downsample_factor), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        if self.first_layer_conv2d:
            x = F.relu(self.bn1(self.conv1(x)))
            x = torch.cat((torch.mean(x, [-1, -2]), torch.std(x, [-1, -2])), dim=-1)
            assert len(x.shape) == 2, x.shape
        else:
            # print(x.shape)
            x = torch.flatten(x, start_dim=1)
            # print(x.shape)
            x = F.relu(self.bn1(self.conv1(x)))  # , kernel_size = 2, stride = 2)
        x = F.relu(self.bn2(self.conv2(x)))  # , kernel_size = 2, stride= 2)
        x = F.relu(self.bn3(self.conv3(x)))  # , kernel_size = 2, stride = 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

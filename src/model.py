import torch.nn.functional as F
from torch import nn
from torch.nn.functional import dropout
from typing import Tuple


class MortalityCNN(nn.Module):
    def __init__ (self, input_size: Tuple[int, int, int] = (3, 128, 128)):
        super(MortalityCNN, self).__init__()

        self.downsample_factor: int = 8  # Factor of 8 bc we use 3 layers of max pooling. I.e. 2**3 = 8
        self.input_size = input_size
        assert self.input_size[1] % self.downsample_factor == 0
        assert self.input_size[2] % self.downsample_factor == 0

        # cnn(in_channel, out_channel, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size= 3, padding= 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 3, padding= 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size= 3, padding= 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * int(self.input_size[1] / self.downsample_factor) *  int(self.input_size[2] / self.downsample_factor), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size = 2, stride = 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size = 2, stride= 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size = 2, stride = 2)

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


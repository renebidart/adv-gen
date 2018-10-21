import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SpatialTopK import SpatialTopK

class SimpleNetMNIST(nn.Module):
    """ No padding, on the input so 4x4. size = 28*28"""
    def __init__(self, num_filters=20):
        super(SimpleNetMNIST, self).__init__()
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, stride=2,  padding=2)
        self.conv2 = nn.Conv2d(15, num_filters, kernel_size=5, stride=2,  padding=2)
        self.fc1 = nn.Linear(num_filters*7*7, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_filters*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def encode_feat(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class TopkNetMNIST(nn.Module):
    """ No padding, on the input so 4x4. size = 28*28"""
    def __init__(self, num_filters=20, topk_num=10):
        super(TopkNetMNIST, self).__init__()
        self.num_filters = num_filters
        self.topk_num = topk_num

        self.topk_layer1 = SpatialTopK(topk=topk_num, frac= None, groups=1)
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, stride=2,  padding=2)
        self.conv2 = nn.Conv2d(15, num_filters, kernel_size=5, stride=2,  padding=2)
        self.fc1 = nn.Linear(num_filters*7*7, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.topk_layer1(self.conv2(x))
        x = x.view(-1, self.num_filters*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def encode_feat(self, x):
        x = F.relu(self.conv1(x))
        x = self.topk_layer1(self.conv2(x))
        return x

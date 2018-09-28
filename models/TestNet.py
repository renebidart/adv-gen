'''
Only to test different activations

Net is composed of resnet blocks of variable activations with variable activations between blocks
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# def get_activation(activation, frac=.25, groups=1):
#     if (activation == 'relu'):
#         activation = nn.ReLU()
#     elif (activation == 'topk'):
#         activation = SpatialTopK(topk=1, frac=frac, groups=groups)
#     else:
#         print('No Activation')
#         activation = None
#     return activation


class Block(nn.Module):
    '''Pre-activation version of the BasicBlock.

    First conv uses the stride, rest are stride 1
    same number of features for each layer
    Shortcut: Uses 1x1 conv with stride to adjust for more layers or a stride

    bn->act->conv->bn->act->conv ...
    '''

    def __init__(self, in_planes, planes, stride, num_layers, use_residual=True, activation=None, frac=.25, groups=1):
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv_all = []
        for i in range(1, num_layers):
            self.conv_all.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        def get_activation(activation, frac=.25, groups=1):
            if (activation == 'relu'):
                activation  = nn.ReLU()
            elif (activation == 'topk'):
                activation = SpatialTopK(topk=1, frac=frac, groups=groups)
            else:
                print('No Activation')
                activation = None
            return activation

        self.use_residual =use_residual
        if activation:
            self.activation = get_activation(activation)

    def forward(self, x):
        out_pre_conv = self.bn1(x)
        out_pre_conv = self.activation(out_pre_conv) if hasattr(self, 'activation') else out_1
        out = self.conv1(out_pre_conv)

        for key, conv_layer in enumerate(self.conv_all):
            out = self.bn2(out)
            out = self.activation(out) if self.activation else x
            out = conv_layer(out)

        if self.use_residual:
            shortcut = self.shortcut(out_pre_conv) if hasattr(self, 'shortcut') else x
            out += shortcut
        return out



class TestNet(nn.Module):
    def __init__(self, block_sizes, block_features, use_residual=True, block_activation=None, 
                 activation=None, frac=.25, groups=1, num_classes=10):
        super(TestNet, self).__init__()

        self.layers = []
        self.activations = []

        def get_activation(activation, frac=.25, groups=1):
            if (activation == 'relu'):
                activation = nn.ReLU()
            elif (activation == 'topk'):
                activation = SpatialTopK(topk=1, frac=frac, groups=groups)
            else:
                print('No Activation')
                activation = None
            return activation

        in_planes = 64
        for i, block_size in enumerate(block_sizes):
            self.layers.append(Block(in_planes, block_features[i], num_layers=block_size, 
                stride=2, use_residual=use_residual, activation=block_activation, frac=frac, groups=groups))
            self.activations.append(get_activation(activation, frac=.25, groups=1))
            in_planes = block_features[i]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activations[i](x) if self.activations[i] else x

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def TestNetNotResNet():
    return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=False, block_activation='relu', 
                 activation=None, frac=.25, groups=1, num_classes=10)

def TestNetMostlyResNet():
    return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, block_activation='relu', 
                 activation=None, frac=.25, groups=1, num_classes=10)

def TestNetResnetTopK(): # .05 is about 91% with normal PResNet
    return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, block_activation='relu', 
                 activation='topk', frac=.05, groups=1, num_classes=10)

def TestNetResnetTopKEverywhere(): # .05 is about 91% with normal PResNet
    return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, block_activation='topk', 
                 activation='topk', frac=.05, groups=1, num_classes=10)

def TestNetResnetTopK_act(): # .05 is about 91% with normal PResNet
    return TestNet(block_sizes=[2,2,2], block_features=[64, 128, 256], use_residual=True, block_activation=None, 
                 activation='topk', frac=.05, groups=1, num_classes=10)

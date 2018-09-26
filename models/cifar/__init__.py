"""
preact_resnet based off this which is based off the original pyotrch one:
Based off this: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

Designed for 32x32 images, not exactly the same as the standard imagenet models.
"""

from .resnet import *
from .preact_resnet import PreActResNet
from .p_resnet_reg import PResNetReg
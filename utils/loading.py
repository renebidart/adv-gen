"""
???
Delete all the loading for models that are never used (all except preact)
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from foolbox.attacks import FGSM, SinglePixelAttack, BoundaryAttack, LBFGSAttack, ProjectedGradientDescent
from models.cifar import PreActResNet

def load_net_cifar(model_loc):
    """ Make a model
    Network must be saved in the form model_name-depth, where this is a unique identifier
    """
    model_file = Path(model_loc).name.rsplit('_')[0]
    model_name = model_file.split('-')[0]
    print('Loading model_file', model_file)
    if (model_name == 'vggnet'):
        model = VGG(int(model_file.split('-')[1]), 10)
    elif (model_name == 'resnet'):
        model = ResNet(int(model_file.split('-')[1]), 10)
    elif (model_name == 'preact_resnet'):
        model = PreActResNet(int(model_file.split('-')[1]), 10)
    elif (model_name == 'wide'):
        model = Wide_ResNet(model_file.split('-')[2][0:2], model_file.split('-')[2][2:4], 0, 10, 32)
    else:
        print('Error : Network should be either [VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)
    model.load_state_dict(torch.load(model_loc)['state_dict'])
    return model


# Return network & a unique file name
def net_from_args(args, num_classes, IM_SIZE):
    if (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes, IM_SIZE)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes, IM_SIZE)
        file_name = 'resnet-'+str(args.depth)
    elif (model_name == 'preact_resnet'):
        model = PreActResNet(args.depth, num_classes)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, IM_SIZE)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [VGGNet / ResNet / PreActResNet/ Wide_ResNet')
        sys.exit(0)
    return net, file_name


def get_attack(attack_type, fmodel):
    if (attack_type == 'FGSM'):
        attack  = foolbox.attacks.FGSM(fmodel)
    elif (attack_type == 'SinglePixelAttack'):
        attack  = foolbox.attacks.SinglePixelAttack(fmodel)
    elif (attack_type == 'boundary'):
        attack  = foolbox.attacks.BoundaryAttack(fmodel)
    elif (attack_type == 'lbfgs'):
        attack  = foolbox.attacks.LBFGSAttack(fmodel)
    elif (attack_type == 'pgd'):
        attack  = foolbox.attacks.ProjectedGradientDescent(fmodel)
    else:
        print('Error: Invalid attack_type')
        sys.exit(0)
    return attack
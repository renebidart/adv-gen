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

import foolbox
from foolbox.attacks import FGSM, SinglePixelAttack, BoundaryAttack, LBFGSAttack, ProjectedGradientDescent
from models.cifar import PreActResNet, PResNetReg, PResNetRegNoRelU
from models.TestNet import TestNetNotResNet, TestNetMostlyResNet
from models.vae import CVAE

def vae_from_args(args):
    if (args.net_type == 'cvae'):
        net = CVAE(num_labels=args.num_labels, latent_size=args.latent_size, img_size=args.IM_SIZE,
                    layer_sizes=args.layer_sizes)
        sizes_str =  "_".join(str(x) for x in args.layer_sizes)
        file_name = 'CVAE-'+str(sizes_str)+'-'+str(args.latent_size)+'-'+str(args.dataset)+'-'+str(args.num_labels)
    else:
        print('Error : Wrong net type')
        sys.exit(0)
    return net, file_name


def load_net_cifar(model_loc):
    """ Make a model
    Network must be saved in the form model_name-depth, where this is a unique identifier
    """
    model_file = Path(model_loc).name
    model_name = model_file.split('-')[0]
    print('Loading model_file', model_file)
    if (model_name == 'vggnet'):
        model = VGG(int(model_file.split('-')[1]), 10)
    elif (model_name == 'resnet'):
        model = ResNet(int(model_file.split('-')[1]), 10)
    # so ugly
    elif (model_name == 'preact_resnet'):
        if model_file.split('/')[-1].split('_')[2] == 'model': 
            model = PreActResNet(int(model_file.split('-')[1].split('_')[0]), 10)
        else:
            model = PResNetReg(int(model_file.split('-')[1]), float(model_file.split('-')[2]), 1, 10)

    elif (model_name == 'wide'):
        model = Wide_ResNet(model_file.split('-')[2][0:2], model_file.split('-')[2][2:4], 0, 10, 32)
    
    # Dumb ones
    elif (model_name == 'PResNetRegNoRelU'):
        model = PResNetRegNoRelU(int(model_file.split('-')[1]), float(model_file.split('-')[2]), 1, 10)
    
    else:
        print(f'Error : {model_file} not found')
        sys.exit(0)
    model.load_state_dict(torch.load(model_loc)['state_dict'])
    return model

def load_net(model_loc):
    model_file = Path(model_loc).name
    model_name = model_file.split('-')[0]
    if (model_name == 'CVAE'):
        model = CVAE(num_labels=int(model_file.split('-')[4].split('_')[0]),
                     latent_size=int(model_file.split('-')[2]), 
                     img_size=32,
                     layer_sizes=[int(i) for i in model_file.split('-')[1].split('_')])
    else:
        print(f'Error : {model_file} not found')
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
    elif (args.net_type == 'preact_resnet'):
        if args.frac != 1:
            net = PResNetReg(args.depth, args.frac, args.groups, num_classes)
            file_name = 'preact_resnet-'+str(args.depth)+'-'+str(args.frac)+'-'+str(args.groups)
        else:
            net = PreActResNet(args.depth, num_classes)
            file_name = 'preact_resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, IM_SIZE)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)

    elif (args.net_type == 'PResNetRegNoRelU'):
        net = PResNetRegNoRelU(args.depth, args.frac, args.groups, num_classes)
        file_name = 'PResNetRegNoRelU-'+str(args.depth)+'-'+str(args.frac)+'-'+str(args.groups)

    elif (args.net_type == 'TestNetNotResNet'):
        net = TestNetNotResNet()
        file_name = 'TestNetNotResNet'
    elif (args.net_type == 'TestNetMostlyResNet'):
        net = TestNetMostlyResNet()
        file_name = 'TestNetMostlyResNet'
    elif (args.net_type == 'TestNetResnetTopK'):
        net = TestNetResnetTopK()
        file_name = 'TestNetResnetTopK'
    elif (args.net_type == 'TestNetResnetTopKEverywhere'):
        net = TestNetResnetTopKEverywhere()
        file_name = 'TestNetResnetTopKEverywhere'
    elif (args.net_type == 'TestNetResnetTopK_act'):
        net = TestNetResnetTopK_act()
        file_name = 'TestNetResnetTopK_act'
    else:
        print('Error : Wrong net type')
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
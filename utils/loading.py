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
from foolbox.attacks import SaltAndPepperNoiseAttack, AdditiveGaussianNoiseAttack, PointwiseAttack

from foolbox.distances import L0, MSE, Linf


from models.cifar import PreActResNet, PResNetReg, PResNetRegNoRelU
from models.TestNet import TestNetNotResNet, TestNetMostlyResNet
from models.cvae import CVAE, CVAE_ABS
from models.vae import VAE
from models.vae_general import VAE_ABS
from models.Nets import SimpleNetMNIST, TopkNetMNIST
from models.FeatureVAE import FEAT_VAE_MNIST

def vae_from_args(args):
    if (args.net_type == 'cvae'):
        net = CVAE(num_labels=args.num_labels, latent_size=args.latent_size, img_size=args.IM_SIZE,
                    layer_sizes=args.layer_sizes)
        sizes_str =  "_".join(str(x) for x in args.layer_sizes)
        file_name = 'CVAE-'+str(sizes_str)+'-'+str(args.latent_size)+'-'+str(args.dataset)+'-'+str(args.num_labels)
    if (args.net_type == 'vae'):
        net = VAE(latent_size=args.latent_size, img_size=args.IM_SIZE, layer_sizes=args.layer_sizes)
        sizes_str =  "_".join(str(x) for x in args.layer_sizes)
        file_name = 'VAE-'+str(sizes_str)+'-'+str(args.latent_size)+'-'+str(args.dataset)
    elif (args.net_type == 'VAE_ABS'):
        net = VAE_ABS(latent_size=args.latent_size, img_size=args.IM_SIZE)
        file_name = 'VAE_ABS-'+str(args.latent_size)+'-'+str(args.dataset)

    elif (args.net_type == 'FEAT_VAE_MNIST'):
        net = FEAT_VAE_MNIST(encoding_model=load_net(args.encoding_model_loc).to(args.device),
                             num_features=args.num_features,
                             latent_size=args.latent_size)
        file_name = 'FEAT_VAE_MNIST-'+str(args.latent_size)+'-'+str(args.num_features)+'-'+str(args.dataset)

    elif (args.net_type == 'CVAE_ABS'):
        net = CVAE_ABS(latent_size=args.latent_size, 
                       img_size=args.IM_SIZE,
                       num_labels=args.num_labels
                       )
        file_name = 'CVAE_ABS-'+str(args.latent_size)+'-'+str(args.dataset)
    else:
        print('Error : Wrong net type')
        sys.exit(0)
    return net, file_name


def load_net(model_loc):
    model_file = Path(model_loc).name
    model_name = model_file.split('-')[0]

    if (model_name == 'CVAE'):
        model = CVAE(num_labels=int(model_file.split('-')[4].split('_')[0]),
                     latent_size=int(model_file.split('-')[2]), 
                     img_size=32,
                     layer_sizes=[int(i) for i in model_file.split('-')[1].split('_')])
    elif (model_name == 'VAE'):
        model = VAE(latent_size=int(model_file.split('-')[2]),
                     img_size=32,
                     layer_sizes=[int(i) for i in model_file.split('-')[1].split('_')])
    elif (model_name == 'VAE_ABS'):
        model = VAE_ABS(latent_size=8, img_size=28)

    elif (model_name == 'CVAE_ABS'):
        model = CVAE_ABS(latent_size=8, img_size=28)

    elif (model_name == 'SimpleNetMNIST'):
        model = SimpleNetMNIST(num_filters=int(model_file.split('-')[1].split('_')[0]))

    elif (model_name == 'TopkNetMNIST'):
        model = TopkNetMNIST(num_filters=int(model_file.split('-')[1].split('_')[0]), 
                        topk_num=int(model_file.split('-')[2].split('_')[0]))

    elif (model_name == 'FEAT_VAE_MNIST'):
        model = VAE_ABS(latent_size=8, img_size=28)

    elif (args.net_type == 'FEAT_VAE_MNIST'):
        model = FEAT_VAE_MNIST(encoding_model=load_net(args.encoding_model_loc).to(args.device),
                             num_features=args.num_features,
                             latent_size=args.latent_size)
        file_name = 'FEAT_VAE_MNIST-'+str(args.latent_size)+'-'+str(args.num_features)+'-'+str(args.dataset)

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
##### Nets for features for Generative classifiers
    elif (args.net_type == 'SimpleNetMNIST'):
        net = SimpleNetMNIST(args.num_filters)
        file_name = 'SimpleNetMNIST-'+str(args.num_filters)
    elif (args.net_type == 'TopkNetMNIST'):
        net = TopkNetMNIST(num_filters=args.num_filters, topk_num=args.topk_num)
        file_name = 'TopkNetMNIST-'+str(args.num_filters)+'-'+str(args.topk_num)

    else:
        print('Error : Wrong net type')
        sys.exit(0)
    return net, file_name


def get_attack(attack_type, fmodel, distance='L0'):
    if distance == 'L0':
        distance = L0
    elif distance == 'MSE':
        distance = MSE
    elif distance == 'Linf':
        distance = Linf
    else:
        print('INVALID DISTANCE!!!')

    if (attack_type == 'FGSM'):
        attack  = foolbox.attacks.FGSM(fmodel, distance=distance)
    elif (attack_type == 'SinglePixelAttack'):
        attack  = foolbox.attacks.SinglePixelAttack(fmodel, distance=distance)
    elif (attack_type == 'boundary'):
        attack  = foolbox.attacks.BoundaryAttack(fmodel, distance=distance)
    elif (attack_type == 'lbfgs'):
        attack  = foolbox.attacks.LBFGSAttack(fmodel, distance=distance)
    elif (attack_type == 'pgd'):
        attack  = foolbox.attacks.ProjectedGradientDescent(fmodel, distance=distance)
    elif (attack_type == 'saltpepper'):
        attack  = foolbox.attacks.SaltAndPepperNoiseAttack(fmodel, distance=distance)
    elif (attack_type == 'gaussian'):
        attack  = foolbox.attacks.AdditiveGaussianNoiseAttack(fmodel, distance=distance)
    elif (attack_type == 'pointwise'):
        attack  = foolbox.attacks.PointwiseAttack(fmodel, distance=distance)
    else:
        print('Error: Invalid attack_type')
        sys.exit(0)
    return attack


### Delete below?

# def load_net_cifar(model_loc):
#     """ Make a model
#     Network must be saved in the form model_name-depth, where this is a unique identifier
#     """
#     model_file = Path(model_loc).name
#     model_name = model_file.split('-')[0]
#     print('Loading model_file', model_file)
#     if (model_name == 'vggnet'):
#         model = VGG(int(model_file.split('-')[1]), 10)
#     elif (model_name == 'resnet'):
#         model = ResNet(int(model_file.split('-')[1]), 10)
#     # so ugly
#     elif (model_name == 'preact_resnet'):
#         if model_file.split('/')[-1].split('_')[2] == 'model': 
#             model = PreActResNet(int(model_file.split('-')[1].split('_')[0]), 10)
#         else:
#             model = PResNetReg(int(model_file.split('-')[1]), float(model_file.split('-')[2]), 1, 10)

#     elif (model_name == 'wide'):
#         model = Wide_ResNet(model_file.split('-')[2][0:2], model_file.split('-')[2][2:4], 0, 10, 32)
    
#     # Dumb ones
#     elif (model_name == 'PResNetRegNoRelU'):
#         model = PResNetRegNoRelU(int(model_file.split('-')[1]), float(model_file.split('-')[2]), 1, 10)
    
#     else:
#         print(f'Error : {model_file} not found')
#         sys.exit(0)
#     model.load_state_dict(torch.load(model_loc)['state_dict'])
#     return model
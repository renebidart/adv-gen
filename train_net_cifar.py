"""
Training for baselines for cifar. 
Use nets from https://github.com/meliketoy/wide-resnet.pytorch (lenet, vgg, resnet, wide-resnet)
Not optimal way to train, using dumb step lr, etc.

"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler

from models.cifar import resnet
from utils.data import make_gen_std_cifar
from utils.train_val import train_model
from utils.loading import net_from_args


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--DATA_PATH', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--device', type=str)

# Defining the network:
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0, type=float, help='dropout_rate')
# training params
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--epochs', default=350, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()


def main(args):
    epochs, batch_size, lr, num_workers = int(args.epochs), int(args.batch_size), float(args.lr),  int(args.num_workers)
    device = torch.device(args.device)

    PATH = Path(args.DATA_PATH)
    SAVE_PATH = Path(args.SAVE_PATH)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Make generators:
    dataloaders = make_gen_std_cifar(PATH, batch_size, num_workers)

    # get the network
    model, model_name = net_from_args(args, num_classes=10, IM_SIZE=32)
    model = model.to(device)
    print(f'--------- Training: {model_name} with depth {args.depth} ---------')

    # get training parameters and train:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.1) # close enough
    
    metrics = {}
    metrics['train_top1_acc'] = []
    metrics['train_losses'] = []
    metrics['val_top1_acc'] = []
    metrics['val_losses'] = []
    best_val_acc = 0
    
    for epoch in range(epochs):
        # train for one epoch
        train_top1_acc, train_losses = train_epoch(dataloaders['train'], model, criterion, optimizer, epoch, device)
        metrics['train_top1_acc'].append(train_top1_acc)
        metrics['train_losses'].append(train_losses)

        # evaluate on validation set
        val_top1_acc, val_losses = validate_epoch(dataloaders['val'], model, criterion, device)
        metrics['val_top1_acc'].append(val_top1_acc)
        metrics['val_losses'].append(val_losses)
        
        scheduler.step()

        # remember best validation accuracy and save checkpoint
        is_best = val_top1_acc > best_val_acc
        best_val_acc = max(val_top1_acc, best_val_acc)
        save_checkpoint({
            'model_name': model_name,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': best_val_acc,
            'metrics': metrics,
        }, is_best, model_name, SAVE_PATH)
        
    pickle.dump(metrics, open(str(PATH)+'/'+str(model_name)+'_metrics.pkl', "wb"))
    

if __name__ == '__main__':
    main(args)
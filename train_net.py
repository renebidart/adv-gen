"""
Training for baselines for cifar. 
Not optimal way to train, using dumb step lr, etc.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler


from utils.data import make_generators_DF_cifar, make_generators_DF_MNIST
from utils.loading import net_from_args
from utils.train_val import train_epoch, validate_epoch, save_checkpoint
# from utils.train_val import train_model ### Not sure if right to put the entire training step into a function.


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--device', type=str)

# Defining the network:
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=18, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0, type=float, help='dropout_rate')
parser.add_argument('--frac', default=1, type=float, help='frac to reatain in topk')
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--groups', default=1, type= int, help='number of independent topk groups')
parser.add_argument('--topk_num', default=10, type= int, help='num to retain in topk')
parser.add_argument('--num_filters', default=10, type= int, help='num filters in last compression conv layer')

# training params
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--IM_SIZE', default=32, type= int)
args = parser.parse_args()


def main(args):
    with torch.cuda.device(1):
        epochs, batch_size, lr, num_workers = int(args.epochs), int(args.batch_size), float(args.lr),  int(args.num_workers)
        IM_SIZE = int(args.IM_SIZE)
        device = torch.device(args.device)

        SAVE_PATH = Path(args.SAVE_PATH)
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        with open(args.files_df_loc, 'rb') as f:
            files_df = pickle.load(f)

        # Make generators:
        if args.dataset == 'CIFAR10':
            dataloaders = make_generators_DF_cifar(files_df, batch_size, num_workers, size=IM_SIZE, 
                                                    path_colname='path', adv_path_colname=None, return_loc=False)
        elif args.dataset == 'MNIST':
            dataloaders = make_generators_DF_MNIST(files_df, batch_size, num_workers, size=IM_SIZE,
                                                    path_colname='path', adv_path_colname=None, return_loc=False, normalize=True)


        # get the network
        model, model_name = net_from_args(args, num_classes=10, IM_SIZE=32)
        model = model.to(device)
        print(f'--------- Training: {model_name} ---------')

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

        for epoch in tqdm(range(epochs)):
            # train for one epoch
            train_top1_acc, train_losses = train_epoch(dataloaders['train'], model, criterion, optimizer, epoch, device)
            metrics['train_top1_acc'].append(train_top1_acc)
            metrics['train_losses'].append(train_losses)

            # evaluate on validation set
            val_top1_acc, val_losses = validate_epoch(dataloaders['val'], model, device, criterion=None)
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
            
        pickle.dump(metrics, open(str(SAVE_PATH)+'/'+str(model_name)+'_metrics.pkl', "wb"))
        

if __name__ == '__main__':
    main(args)

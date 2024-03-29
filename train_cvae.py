"""
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


from utils.data import make_generators_DF_cifar, make_generators_DF_MNIST
from utils.loading import vae_from_args
from utils.train_val_auto import train_epoch_auto, validate_epoch_auto, save_checkpoint


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--device', type=str)

# Defining the network:
parser.add_argument('--net_type', default='cvae', type=str, help='model')
parser.add_argument('--layer_sizes','--list', nargs='+')
parser.add_argument('--latent_size', default=32, type=int, help='model')
parser.add_argument('--dropout', default=0, type=float, help='dropout_rate')
parser.add_argument('--frac', default=1, type=float, help='frac to reatain in topk')
parser.add_argument('--groups', default=1, type= int, help='number of independent topk groups')
# training params
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_labels', default=10, type=int)
parser.add_argument('--IM_SIZE', default=28, type=int)
args = parser.parse_args()


def main(args):
    # ??? Remove this:
    with torch.cuda.device(1):
        if args.layer_sizes:
            args.layer_sizes = [int(i) for i in args.layer_sizes]
        epochs, batch_size, lr, num_workers = int(args.epochs), int(args.batch_size), float(args.lr),  int(args.num_workers)
        num_labels, IM_SIZE= int(args.num_labels), int(args.IM_SIZE)
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
                                                    path_colname='path', adv_path_colname=None, return_loc=False)

        # get the network
        model, model_name = vae_from_args(args)
        model = model.to(device)
        print('next(model.parameters()).device', next(model.parameters()).device)
        print(f'--------- Training: {model_name} ---------')

        for p in model.parameters():
            p.requires_grad=True

        # get training parameters and train:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/3), gamma=0.3) # close enough
        
        metrics = {}
        metrics['train_losses'] = []
        metrics['val_losses'] = []
        best_val_loss = 1000000

        criterion = model.loss
        
        for epoch in range(epochs):
            # train for one epoch
            train_losses = train_epoch_auto(epoch, lr, dataloaders['train'], model, optimizer, criterion, device)
            metrics['train_losses'].append(train_losses)

            # evaluate on validation set
            val_losses = validate_epoch_auto(dataloaders['val'], model, criterion, device)
            metrics['val_losses'].append(val_losses)
            
            scheduler.step()

            # remember best validation accuracy and save checkpoint
            is_best = val_losses < best_val_loss
            best_val_loss = min(val_losses, best_val_loss)
            save_checkpoint({
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'metrics': metrics,
            }, is_best, model_name, SAVE_PATH)
            
        pickle.dump(metrics, open(str(SAVE_PATH)+'/'+str(model_name)+'_metrics.pkl', "wb"))

if __name__ == '__main__':
    main(args)

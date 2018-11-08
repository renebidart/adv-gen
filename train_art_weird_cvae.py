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


from utils.data import make_generators_DF
from utils.loading import vae_from_args
from utils.train_val_auto import train_epoch_auto, validate_epoch_auto, save_checkpoint 


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--files_dict_loc', type=str)
parser.add_argument('--SAVE_PATH', type=str)
parser.add_argument('--base_path', type=str)
parser.add_argument('--device', type=str)

# Defining the network:
parser.add_argument('--small_net_type', default='CVAE_SMALL', type=str)
parser.add_argument('--net_type', default='ConvVAE2d', type=str)
parser.add_argument('--cvae_input_sz', default=16, type=int)
parser.add_argument('--stride', default=2, type=int)
parser.add_argument('--latent_size', default=16, type=int)
parser.add_argument('--latent_size_small_vae', default=16, type=int)

# training params
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_labels', default=11, type=int)
parser.add_argument('--IM_SIZE', default=64, type=int)
args = parser.parse_args()


def main(args):
    device = torch.device(args.device)
    base_path = str(args.base_path)
    epochs, batch_size, lr, num_workers = int(args.epochs), int(args.batch_size), float(args.lr),  int(args.num_workers)
    num_labels, IM_SIZE= int(args.num_labels), int(args.IM_SIZE)

    SAVE_PATH = Path(args.SAVE_PATH)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with open(args.files_dict_loc, 'rb') as f:
        files_dict = pickle.load(f)

    # Make generators:
    dataloaders = make_generators_DF(files_dict=files_dict, base_path=base_path, batch_size=batch_size, IM_SIZE=IM_SIZE, 
                                     select_label=None, return_loc=False, path_colname='path', label_colname='label', num_workers=4)

    # get the network
    model, model_name = vae_from_args(args)
    model = model.to(device)
    print('next(model.parameters()).device', next(model.parameters()).device)
    print(f'--------- Training: {model_name} ---------')

    for p in model.parameters():
        p.requires_grad=True

    # get training parameters and train:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs/2), gamma=0.2) # close enough
    
    metrics = {}
    metrics['train_losses'] = []
    metrics['val_losses'] = []
    best_val_loss = 100000000

    criterion = model.loss
    
    for epoch in tqdm(range(epochs)):
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

    pickle.dump(metrics, open(str(SAVE_PATH)+'/'+str(model_name)+'_stride_'+str(args.stride)+'_metrics.pkl', "wb"))

if __name__ == '__main__':
    with torch.cuda.device(torch.device(args.device).index):
        main(args)

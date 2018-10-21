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


from utils.data import make_generators_DF_MNIST
from utils.loading import load_net
from utils.evaluation import eval_gen_vae, eval_gen_cvae
from models.generative_classify import optimize_latent_cvae, gen_classify_cvae, optimize_latent, gen_classify

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--model_loc', type=str) # vae, cvae
parser.add_argument('--model_type', type=str) # vae, cvae

parser.add_argument('--device', type=str)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--num_times', type=int, default=100)
parser.add_argument('--iterations', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=16)
parser.add_argument('--df_sample_num', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--IM_SIZE', default=32, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--batch_size', default=1, type=int)
args = parser.parse_args()


def main(args):
    with torch.cuda.device(1): # ??? Remove this:
        labels = list(range(10))

        batch_size, lr, num_workers, latent_size = int(args.batch_size), float(args.lr),  int(args.num_workers), int(args.latent_size)
        IM_SIZE, model_loc, num_times, iterations = int(args.IM_SIZE), Path(args.model_loc), int(args.num_times), int(args.iterations)
        device = torch.device(args.device)

        with open(args.files_df_loc, 'rb') as f:
            files_df = pickle.load(f)
            files_df['val'] = files_df['val'].sample(n=int(args.df_sample_num))

        if args.dataset == 'CIFAR10':
            dataloaders = make_generators_DF_cifar(files_df, batch_size, num_workers, size=IM_SIZE, 
                                                    path_colname='path', adv_path_colname=None, label=None, return_loc=True)
        elif args.dataset == 'MNIST':
            dataloaders = make_generators_DF_MNIST(files_df, batch_size, num_workers, size=IM_SIZE,
                                                    path_colname='path', adv_path_colname=None, label=None, return_loc=True, normalize=True)

        # Train for each of the labels, here the model_loc is not an actual loc, just the base
        if args.model_type == 'vae':
            model_dict = {}
            model_file = Path(model_loc).name
            model_name = model_file.split('-')[0]
            for label in labels:
                model_file_cr = model_file + '_label_'+str(label)+'_model_best.pth.tar'
                model_loc_cr = str(model_loc.parent / model_file_cr)
                model_dict[label] = load_net(model_loc_cr).to(device).eval()

            results = eval_gen_vae(dataloaders['val'], model_dict, device, 
                                   num_times=num_times, iterations=iterations, latent_size=latent_size)

        elif args.model_type == 'cvae':
            model = load_net(model_loc).to(device).eval()
            results = eval_gen_cvae(dataloaders['val'], model, labels, device, 
                                    num_times=num_times, iterations=iterations, latent_size=latent_size)


    SAVE_PATH = (str(model_loc).rsplit('-', 1)[:-1][0]+'_iter'+str(iterations)+
                '_nt'+str(num_times)+'_nsamp'+str(args.df_sample_num)+'_results.pkl')
    pickle.dump(results, open(str(SAVE_PATH), "wb"))

if __name__ == '__main__':
    main(args)

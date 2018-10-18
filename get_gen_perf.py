""" 
USE THIS INSTEAD OF eval_gen_classifier
Proper way to predict using generative models, use new gen model class instead of hacky shit in "eval_gen_classifier"

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

from utils.data import make_generators_DF_MNIST
from utils.loading import load_net
from models.gen_classifiers import GenerativeCVAE, GenerativeVAE

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
    batch_size, lr, num_workers, latent_size = int(args.batch_size), float(args.lr),  int(args.num_workers), int(args.latent_size)
    IM_SIZE, model_loc, num_times, iterations = int(args.IM_SIZE), Path(args.model_loc), int(args.num_times), int(args.iterations)
    device = torch.device(args.device)

    num_workers = 2 ## do 0 if doesnt work
    num_classes = 10

    labels = list(range(num_classes))
    all_results = pd.DataFrame()

    # Get file sample and make generators:
    with open(args.files_df_loc, 'rb') as f:
        files_df = pickle.load(f)
        files_df['val'] = files_df['val'].sample(n=int(args.df_sample_num))

    if args.dataset == 'CIFAR10':
        dataloaders = make_generators_DF_cifar(files_df, batch_size, num_workers, size=IM_SIZE, 
                                                path_colname='path', adv_path_colname=None, return_loc=True)
    elif args.dataset == 'MNIST':
        dataloaders = make_generators_DF_MNIST(files_df, batch_size, num_workers, size=IM_SIZE,
                                                path_colname='path', adv_path_colname=None, return_loc=True, normalize=True)

    # Train for each of the labels, here the model_loc is not an actual loc, just the base
    if args.model_type == 'vae':
        model_dict = {}
        model_file = Path(model_loc).name
        model_name = model_file.split('-')[0]
        for label in labels:
            model_file_cr = model_file + '_label_'+str(label)+'_model_best.pth.tar'
            model_loc_cr = str(model_loc.parent / model_file_cr)
            model_dict[label] = load_net(model_loc_cr).to(device).eval()

        gen_model = GenerativeVAE(model_dict=model_dict, labels=labels, latent_size=latent_size, device=device)

        with torch.no_grad():
            for i, (tensor_img, label, path) in enumerate(tqdm(dataloaders['val'])):
                tensor_img, label, file_loc = tensor_img.to(device), label.to(device), path[0]
                pred = np.argmax(gen_model(tensor_img, iterations=iterations, num_times=num_times).cpu().numpy())

                all_results = all_results.append({'path': path, 
                                                  'true_label': int(label.cpu().numpy()),
                                                  'predicted_label': int(pred),
                                                 }, ignore_index=True)

    # If CVAE, load model and predict normally:
    elif args.model_type == 'cvae':
        model = load_net(model_loc).to(device).eval()
        gen_model = GenerativeCVAE(model=model, labels=labels, latent_size=latent_size, device=device).eval()
        all_results = pd.DataFrame()

        with torch.no_grad():
            for i, (tensor_img, label, path) in enumerate(tqdm(dataloaders['val'])):
                tensor_img, label, file_loc = tensor_img.to(device), label.to(device), path[0]
                pred = np.argmax(gen_model(tensor_img, iterations=iterations, num_times=num_times).cpu().numpy())

                all_results = all_results.append({'path': path, 
                                                  'true_label': int(label.cpu().numpy()),
                                                  'predicted_label': int(pred), 
                                                 }, ignore_index=True)
            

    SAVE_PATH = (str(model_loc).rsplit('-', 1)[:-1][0]+'_iter'+str(iterations)+
                '_nt'+str(num_times)+'_nsamp'+str(args.df_sample_num)+'_results.pkl')
    print(f'saving at: {SAVE_PATH}')
    pickle.dump(all_results, open(str(SAVE_PATH), "wb"))


if __name__ == '__main__':
    with torch.cuda.device(torch.device(args.device).index): # ??? Remove this:
        main(args)

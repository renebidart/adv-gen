import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import foolbox

from utils.loading import load_net, get_attack
from utils.data import make_generators_DF_MNIST
from models.gen_classifiers import GenerativeCVAE
# from utils.display import load_image

parser = argparse.ArgumentParser()
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--model_loc', type=str)
parser.add_argument('--attack_type', type=str)
parser.add_argument('--sample_num', type=int)
parser.add_argument('--IM_SIZE', type=int)
parser.add_argument('--dataset', default='cvae', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--device', type=str)
args = parser.parse_args()

files_df_loc, model_loc, attack_type, device = args.files_df_loc, args.model_loc, args.attack_type, torch.device(args.device)
sample_num, IM_SIZE = int(args.sample_num), int(args.IM_SIZE)
RESULT_PATH = Path(args.model_loc).parent

def get_adv_perf(files_df_loc, RESULT_PATH, model_loc, attack_type, sample_num, device):

    num_workers = 0 ## ??? what?
    batch_size = 1

    num_classes = 10
    labels = list(range(num_classes))

    with open(args.files_df_loc, 'rb') as f:
        files_df = pickle.load(f)
    files_df['val'] = files_df['val'].sample(n=sample_num)

    # Make generators:
    if args.dataset == 'CIFAR10':
        dataloaders = make_generators_DF_cifar(files_df, batch_size, num_workers, size=IM_SIZE, 
                                                path_colname='path', adv_path_colname=None, return_loc=True)
    elif args.dataset == 'MNIST':
        dataloaders = make_generators_DF_MNIST(files_df, batch_size, num_workers, size=IM_SIZE,
                                                path_colname='path', adv_path_colname=None, return_loc=True, normalize=False)

        transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

    model = load_net(model_loc).to(device).eval()
    gen_model = GenerativeCVAE(model=model, labels=labels, latent_size=16, device=device).eval()
    fmodel = foolbox.models.PyTorchModel(gen_model, bounds=(0, 1), num_classes=num_classes, device=device) # no preprocessing since using data loader
    attack  = get_attack(attack_type, fmodel)

    num_failed = 0
    results = pd.DataFrame() 

    for i, batch in enumerate(tqdm(dataloaders['val'])):
        # Foolbox always wants numpy arrays, and we are using single batch, so this batch dim is removed.
        img, label, file_loc = batch[0].to(device), batch[1].to(device), batch[2][0]

        # image, label, file_loc = batch[0].numpy(), int(batch[1].numpy()), batch[2][0]
        # image = load_image(file_loc) See if should use this with the foolbox preprocessign instead
        # np_img = np.expand_dims(np.squeeze(img.numpy()), 0)
        # np_img = np.squeeze(img.cpu().numpy())

        trans_img = transform(img.squeeze().unsqueeze(0).cpu())
        trans_img = trans_img.unsqueeze(0).to(device)
        trans_orig_pred = np.argmax(gen_model(trans_img).cpu().numpy())

        np_img = np.expand_dims(np.squeeze(img.cpu().numpy()), 0)
        np_label = int(label.cpu().numpy()[0])
        np_adv_img = attack(np_img, np_label)

        adv_img = torch.from_numpy(np_adv_img).float().unsqueeze(0).to(device)
        orig_pred = np.argmax(gen_model(img).cpu().numpy())
        adv_pred = np.argmax(gen_model(adv_img).cpu().numpy())

        # np_trans_img = np.expand_dims(np.squeeze(img.cpu().numpy()), 0)
        # trans_adv_img = attack(trans_img, np_label)
        # trans_adv_pred = np.argmax(gen_model(trans_adv_img).cpu().numpy())

        f_pred = np.argmax(fmodel.predictions(np_img))
        f_adv_pred = np.argmax(fmodel.predictions(np_adv_img))

        print(f'FOOLBOX: orig_pred: {f_pred}, adv_pred: {f_adv_pred}')
        print(f'Normal Model, Trans: orig_pred: {trans_orig_pred}, ')#adv_pred: {trans_adv_pred}')
        print(f'Normal Model, None: orig_pred: {orig_pred}, adv_pred: {adv_pred}')
        print(f'Original Label {np_label}')


        if adv_img is None:
            print('adv_img is None')
            num_failed +=1

        results = results.append({'path': file_loc, 'true_label': np.squeeze(label.item()),
                                  'orig_pred': np.squeeze(orig_pred), 'adv_pred': np.squeeze(adv_pred), 
                                  'trans_orig_pred': np.squeeze(trans_orig_pred), 
                                  'f_pred': f_pred, 'f_adv_pred': f_adv_pred, 
                                   'attack_type': attack_type}, ignore_index=True)

    with open(str(RESULT_PATH)+'/results_ns'+str(sample_num)+'_'+str(attack_type)+'_adv.pkl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(f'Finished testing {sample_num} images. Number of failures: {num_failed}')


if __name__ == '__main__':
    get_adv_perf(files_df_loc, RESULT_PATH, model_loc, attack_type, sample_num, device)

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
from models.gen_classifiers import GenerativeCVAE, GenerativeVAE
# from utils.display import load_image

parser = argparse.ArgumentParser()
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--model_loc', type=str)
parser.add_argument('--model_type', type=str) # vae, cvae

parser.add_argument('--attack_type', type=str)
parser.add_argument('--distance', default='MSE', type=str)
parser.add_argument('--sample_num', type=int)
parser.add_argument('--IM_SIZE', type=int)
parser.add_argument('--latent_size', type=int)
parser.add_argument('--dataset', default='cvae', type=str)
parser.add_argument('--device', type=str)
args = parser.parse_args()

files_df_loc, attack_type, model_type = args.files_df_loc, args.attack_type, args.model_type
sample_num, IM_SIZE, latent_size= int(args.sample_num), int(args.IM_SIZE), int(args.latent_size)
model_loc = Path(args.model_loc)
RESULT_PATH = Path(args.model_loc).parent
device = torch.device(args.device)
distance = str(args.distance)


def get_adv_perf(files_df_loc, RESULT_PATH, model_loc, model_type, attack_type, distance, sample_num, device):
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
                                                path_colname='path', adv_path_colname=None, return_loc=True, normalize=False)
    elif args.dataset == 'MNIST':
        dataloaders = make_generators_DF_MNIST(files_df, batch_size, num_workers, size=IM_SIZE,
                                                path_colname='path', adv_path_colname=None, return_loc=True, normalize=False)
    else:
        print("Incorrect Dataset")

        # transform = transforms.Compose([
        #                         transforms.ToPILImage(),
        #                         transforms.Resize(32),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize((0.1307,), (0.3081,))])
    mean = 0.1307  #np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = 0.3081  #np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

    if args.model_type == 'vae': # then the model_loc isn't the real location
        model_dict = {}
        model_file = Path(model_loc).name
        model_name = model_file.split('-')[0]
        for label in labels:
            model_file_cr = model_file + '_label_'+str(label)+'_model_best.pth.tar'
            model_loc_cr = str(model_loc.parent / model_file_cr)
            model_dict[label] = load_net(model_loc_cr).to(device).eval()

        gen_model = GenerativeVAE(model_dict=model_dict, labels=labels, latent_size=latent_size, device=device)

    elif args.model_type == 'cvae':
        model = load_net(model_loc).to(device).eval()
        gen_model = GenerativeCVAE(model=model, labels=labels, latent_size=16, device=device).eval()

    else:
        print("Incorrect Model Type")

    fmodel = foolbox.models.PyTorchModel(gen_model, bounds=(0, 1), preprocessing=(mean, std), 
                                         num_classes=num_classes, device=device)
    attack  = get_attack(attack_type, fmodel, distance=distance)

    num_failed = 0
    results = pd.DataFrame() 

    for i, batch in enumerate(tqdm(dataloaders['val'])):
        # Foolbox always wants numpy arrays, and we are using single batch, so this batch dim is removed.
        img, label, file_loc = batch[0].to(device), batch[1].to(device), batch[2][0]

        # image, label, file_loc = batch[0].numpy(), int(batch[1].numpy()), batch[2][0]
        # image = load_image(file_loc) See if should use this with the foolbox preprocessign instead
        # np_img = np.expand_dims(np.squeeze(img.numpy()), 0)
        # np_img = np.squeeze(img.cpu().numpy())

        # trans_img = transform(img.squeeze().unsqueeze(0).cpu())
        # trans_img = trans_img.unsqueeze(0).to(device)
        # trans_orig_pred = np.argmax(gen_model(trans_img).cpu().numpy())

        print('torch.min(img)', torch.min(img))
        print('torch.max(img)', torch.max(img))

        np_img = np.expand_dims(np.squeeze(img.cpu().numpy()), 0)
        np_label = int(label.cpu().numpy()[0])
        print('np.amin(np_img)', np.amin(np_img))
        print('np.amax(np_img)', np.amax(np_img))
        print('original prediction foolbox: ', np.argmax(fmodel.predictions(np_img)))

        adv_object = attack(np_img, np_label, unpack=False)
        adv_distance = adv_object.distance
        np_adv_img = adv_object.image

        print('adv_object.output', adv_object.output)
        print('adv_distance', adv_distance)
        print('np.amin(np_adv_img)', np.amin(np_adv_img))
        print('np.amax(np_adv_img)', np.amax(np_adv_img))

        # adv_img = torch.from_numpy(np_adv_img).float().unsqueeze(0).to(device)
        # orig_pred = np.argmax(gen_model(img).cpu().numpy())
        # adv_pred = np.argmax(gen_model(adv_img).cpu().numpy())

        # np_trans_img = np.expand_dims(np.squeeze(img.cpu().numpy()), 0)
        # trans_adv_img = attack(trans_img, np_label)
        # trans_adv_pred = np.argmax(gen_model(trans_adv_img).cpu().numpy())

        f_orig_pred = np.argmax(fmodel.predictions(np_img))
        f_adv_pred = np.argmax(fmodel.predictions(np_adv_img))

        print(f'FOOLBOX: orig_pred: {f_orig_pred}, adv_pred: {f_adv_pred}')
        # print(f'Normal Model, Trans: orig_pred: {trans_orig_pred}, ')#adv_pred: {trans_adv_pred}')
        # print(f'Normal Model, None: orig_pred: {orig_pred}, adv_pred: {adv_pred}')
        print(f'Original Label {np_label}')

        if np_adv_img is None:
            print('-----np_adv_img is None, Attack Failed')
            num_failed +=1

        results = results.append({'path': file_loc, 'true_label': np.squeeze(label.item()), 'adv_distance': adv_distance.value,
                                  # 'orig_pred': np.squeeze(orig_pred), 'adv_pred': np.squeeze(adv_pred), 
                                  # 'trans_orig_pred': np.squeeze(trans_orig_pred), 
                                  'f_orig_pred': f_orig_pred, 'f_adv_pred': f_adv_pred, 
                                   'attack_type': attack_type, 'distance': distance}, ignore_index=True)

    save_loc = str(RESULT_PATH)+'/results_'+str(Path(model_loc).name)+'_ns'+str(sample_num)+'_'+str(attack_type)+'_'+str(distance)+'_adv.pkl'
    print(f'saving at: {save_loc}')
    with open(save_loc, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(f'Finished testing {sample_num} images. Number of failures: {num_failed}')


if __name__ == '__main__':
    with torch.cuda.device(device.index): # ??? Remove this:
        get_adv_perf(files_df_loc, RESULT_PATH, model_loc, model_type, attack_type, distance, sample_num, device)

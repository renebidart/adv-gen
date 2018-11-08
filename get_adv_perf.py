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
from models.gen_classifiers import GenerativeCVAE, GenerativeVAE, GenerativeFeatVAE


import foolbox 
from foolbox import attacks as fa

parser = argparse.ArgumentParser()
parser.add_argument('--files_df_loc', type=str)
parser.add_argument('--model_loc', type=str)
parser.add_argument('--encoding_model_loc', type=str)
parser.add_argument('--model_type', type=str) # vae, cvae

parser.add_argument('--attack_type', type=str)
parser.add_argument('--distance', default='MSE', type=str)
parser.add_argument('--sample_num', type=int)
parser.add_argument('--IM_SIZE', type=int)
parser.add_argument('--latent_size', type=int)
parser.add_argument('--dataset', default='cvae', type=str)
parser.add_argument('--device', type=str)
args = parser.parse_args()


def get_adv_perf(args):
    sample_num, IM_SIZE, latent_size= int(args.sample_num), int(args.IM_SIZE), int(args.latent_size)
    model_loc = Path(args.model_loc)
    RESULT_PATH = Path(args.model_loc).parent
    device = torch.device(args.device)
    distance = str(args.distance)

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

        mean = 0.1307
        std = 0.3081
        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(IM_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((mean,), (std,))])
    else:
        print("Incorrect Dataset")


    # if args.model_type == 'vae': # then the model_loc isn't the real location
    #     model_dict = {}
    #     model_file = Path(model_loc).name
    #     model_name = model_file.split('-')[0]
    #     for label in labels:
    #         model_file_cr = model_file + '_label_'+str(label)+'_model_best.pth.tar'
    #         model_loc_cr = str(model_loc.parent / model_file_cr)
    #         model_dict[label] = load_net(model_loc_cr).to(device).eval()

    #     gen_model = GenerativeVAE(model_dict=model_dict, labels=labels, latent_size=latent_size, device=device).eval()

    # elif args.model_type == 'cvae':
    #     model = load_net(model_loc).to(device).eval()
    #     gen_model = GenerativeCVAE(model=model, labels=labels, latent_size=latent_size, device=device).eval()
    # Train for each of the labels, here the model_loc is not an actual loc, just the base

    if args.model_type == 'vae' or args.model_type == 'feat_vae':
        model_dict = {}
        model_file = Path(model_loc).name
        model_name = model_file.split('-')[0]
        for label in labels:
            model_file_cr = model_file + '_label_'+str(label)+'_model_best.pth.tar'
            model_loc_cr = str(model_loc.parent / model_file_cr)
            model_dict[label] = load_net(model_loc_cr, args).to(device).eval()

        print('args.model_type', args.model_type)
        
        if args.model_type == 'vae':
            print(f'Loading VAE')
            gen_model = GenerativeVAE(model_dict=model_dict, labels=labels, latent_size=latent_size, device=device).to(device).eval()
        elif args.model_type == 'feat_vae':
            print(f'Loading FEATURE VAE')
            gen_model = GenerativeFeatVAE(model_dict=model_dict, labels=labels, latent_size=latent_size, device=device).to(device).eval()
        else:
            print("Invalid model_type")

    # If CVAE, load model and predict normally:
    elif args.model_type == 'cvae':
        print(f'Loading CVAE')
        model = load_net(model_loc, args).to(device).eval()
        gen_model = GenerativeCVAE(model=model, labels=labels, latent_size=latent_size, device=device).to(device).eval()
    else:
        print("Incorrect Model Type")

    # ATTACK:
    fmodel = foolbox.models.PyTorchModel(gen_model, bounds=(0, 1), preprocessing=(mean, std), 
                                         num_classes=num_classes, device=device)
    # attack  = get_attack(args.attack_type, fmodel, distance=distance)


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

        np_img = np.expand_dims(np.squeeze(img.cpu().numpy()), 0)
        np_label = int(label.cpu().numpy()[0])
        # print('original prediction foolbox: ', np.argmax(fmodel.predictions(np_img)))

        # if args.attack_type == 'boundary':
        #     # adv_object = attack(np_img, np_label, unpack=False, tune_batch_size=False, verbose=True, log_every_n_steps=1)
        # else:
        #     adv_object = attack(np_img, np_label, unpack=False)

        # $$$$$$$$$$$$$$$$$$

        att = fa.PointwiseAttack(fmodel)
        metric = foolbox.distances.L0
        criterion = foolbox.criteria.Misclassification()

        # Estimate gradients from scores
        if not gen_model.has_grad: 
            GE = foolbox.gradient_estimators.CoordinateWiseGradientEstimator(0.1)
            fmodel = foolbox.models.ModelWithEstimatedGradients(fmodel, GE)


        # gernate Adversarial
        a = foolbox.adversarial.Adversarial(fmodel, criterion, np_img, np_label, distance=metric)
        att(a)

        print('pred', np.argmax(fmodel.predictions(a.image)))
        print('distance', a.normalized_distance)



        # $$$$$$$$$$$$$$$$$$


        adv_distance = adv_object.distance
        np_adv_img = adv_object.image

        # adv_img = torch.from_numpy(np_adv_img).float().unsqueeze(0).to(device)
        # orig_pred = np.argmax(gen_model(img).cpu().numpy())
        # adv_pred = np.argmax(gen_model(adv_img).cpu().numpy())

        # np_trans_img = np.expand_dims(np.squeeze(img.cpu().numpy()), 0)
        # trans_adv_img = attack(trans_img, np_label)
        # trans_adv_pred = np.argmax(gen_model(trans_adv_img).cpu().numpy())

        f_orig_pred = np.argmax(fmodel.predictions(np_img))
        f_adv_pred = np.argmax(fmodel.predictions(np_adv_img))

        # print(f'FOOLBOX: orig_pred: {f_orig_pred}, adv_pred: {f_adv_pred}')
        # print(f'Original Label {np_label}')

        if np_adv_img is None:
            print('-----np_adv_img is None, Attack Failed')
            num_failed +=1

        results = results.append({'path': file_loc, 'true_label': np.squeeze(label.item()), 'adv_distance': adv_distance.value,
                                  'f_orig_pred': f_orig_pred, 'f_adv_pred': f_adv_pred, 
                                   'attack_type': attack_type, 'distance': distance}, ignore_index=True)

    save_loc = str(RESULT_PATH)+'/results_'+str(Path(model_loc).name)+'_ns'+str(sample_num)+'_'+str(attack_type)+'_'+str(distance)+'_adv.pkl'
    print(f'saving at: {save_loc}')
    with open(save_loc, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print(f'Finished testing {sample_num} images. Number of failures: {num_failed}')


if __name__ == '__main__':
    with torch.cuda.device(torch.device(args.device).index): # ??? Remove this:
        get_adv_perf(args)

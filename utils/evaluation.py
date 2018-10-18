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

from utils.data import make_generators_DF_cifar
from models.generative_classify import optimize_latent_cvae, gen_classify_cvae, optimize_latent, gen_classify


def eval_gen_cvae(dataloader, model, labels, device, num_times=50, iterations=50, latent_size=16):
    model.eval()
    all_results = pd.DataFrame()
    
    with torch.no_grad():
        for i, (tensor_img, label, path) in enumerate(tqdm(dataloader)):
            path = path[0]
            
            tensor_img = tensor_img.to(device)
            results, predicted_label = gen_classify_cvae(tensor_img, labels, model, num_times=num_times, 
                                                         iterations=iterations, latent_size=latent_size, 
                                                         device=device, KLD_weight=1)
                        
            for i, true_label in enumerate(label): 
                all_results = all_results.append({'path': path, 
                                                  'true_label': int(true_label.cpu().numpy()),
                                                  'predicted_label': int(predicted_label), 
                                                 }, ignore_index=True)
    return all_results



def eval_gen_vae(dataloader, model_dict, device, num_times=50, iterations=50, latent_size=16):
    all_results = pd.DataFrame()
    
    with torch.no_grad():
        for i, (tensor_img, label, path) in enumerate(tqdm(dataloader)):
            path = path[0]
            
            tensor_img = tensor_img.to(device)
            results, predicted_label = gen_classify(tensor_img, model_dict,
                                          num_times=num_times, iterations=iterations,
                                          latent_size=latent_size, device=device, KLD_weight=1)
            
            
    
            for i, true_label in enumerate(label): 
                all_results = all_results.append({'path': path,
                                                  'true_label': int(true_label.cpu().numpy()),
                                                  'predicted_label': int(predicted_label), 
                                                 }, ignore_index=True)
    return all_results


def evaluate_adv_files_df(files_df, denoise_model, device):
    results = pd.DataFrame()
    
    dataloaders = make_generators_DF_cifar(files_df, batch_size=5, num_workers=4, size=32, 
                                   path_colname='path', adv_path_colname='adv_path', return_loc=True)
    
    denoise_model.eval()

    with torch.no_grad():
        for i, (orig, adv, target, (path, adv_path)) in enumerate(dataloaders['val']):
            orig, adv, target = orig.to(device), adv.to(device), target.to(device)
            orig_out, denoised_orig_pred, adv_out, denoised_adv_pred = denoise_model(orig, adv, eval_mode=True)
            
            for i, true_label in enumerate(target):                
                results = results.append({'path': path[i], 'adv_path': adv_path[i], 
                                          'true_label': int(true_label.cpu().numpy()),
                                          'orig_pred': int(orig_out[i].argmax().cpu().numpy()), 
                                          'denoised_orig_pred': int(denoised_orig_pred[i].argmax().cpu().numpy()),
                                          'adv_pred': int(adv_out[i].argmax().cpu().numpy()), 
                                          'denoised_adv_pred': int(denoised_adv_pred[i].argmax().cpu().numpy())
                                         }, ignore_index=True)
        return results

def get_metrics(results):
    total = len(results)
    orig_acc = len(results[results['true_label']==results['orig_pred']])/total
    adv_acc = len(results[results['true_label']==results['adv_pred']])/total
    denoised_adv_acc = len(results[results['true_label']==results['denoised_adv_pred']])/total
    
    correct_df = results[results['true_label']==results['orig_pred']]
    ibh_correct = len(correct_df[correct_df['true_label']==correct_df['denoised_adv_pred']])/len(correct_df) 

    print(f'Total Observations: {total}')
    print(f'Original Accuracy: {100*orig_acc:.2f}%')
    print(f'Adversarial Accuracy: {100*adv_acc:.2f}%')
    print(f'Denoised Adversarial Accuracy : {100*denoised_adv_acc:.2f}%')
    print(f'Percent of correct remaing correct: {100*ibh_correct:.2f}%')


import os
import sys
import pickle
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.nn import functional as F
import torch.optim as optim

def VAE_test_loss(recon_x, mu, logsigma, x, KLD_weight=1):    
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    loss = BCE+KLD_weight*KLD
    return loss, BCE, KLD

    
def optimize_latent(img, mu, logvar, model, lr=.001, iterations=50, KLD_weight=1):
    """ Do SGD on the latent variables to minimize loss for given model.
    mu, logvar must be proper format leaf variables
    """
    model=model.eval()
    for p in model.parameters():
        p.requires_grad=False

    BCE_list = []
    KLD_list = []

    optimizer = optim.Adam([mu, logvar], lr=lr)

    for i in range(iterations):
        z = model.reparameterize(mu, logvar)
        recon_x = model.decode(z)

        loss, BCE, KLD = VAE_test_loss(recon_x, mu, logvar, img, KLD_weight)
        BCE_list.append(BCE.item())
        KLD_list.append(KLD.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return recon_x, z, loss, BCE_list, KLD_list


def gen_classify(img, model_dict, num_times, iterations, latent_size, device, KLD_weight=1):
    results = {}
    for label, model in model_dict.items():
        results[label] = []

    for trial in range(num_times):
        for label, model in model_dict.items():
            rand_mu = np.random.normal(0,1, (1, latent_size))
            rand_logvar = np.random.normal(0,1, (1, latent_size))
            mu = torch.tensor(rand_mu, device=device, requires_grad=True).type(torch.cuda.FloatTensor)
            logvar = torch.tensor(rand_logvar, device=device, requires_grad=True).type(torch.cuda.FloatTensor)
            mu = Variable(mu.data, requires_grad=True)
            logvar = Variable(logvar.data, requires_grad=True)

            recon_x, z, loss, BCE_list, KLD_list = optimize_latent(img, mu, logvar, model, lr=.001, iterations=iterations, KLD_weight=KLD_weight)

            results[label].append(loss.item())
    mins = {k:np.min(v) for (k,v) in results.items()}
    predicted_label = min(mins, key=mins.get)
    return results, predicted_label


def optimize_latent_cvae(imgs, labels, mu, logvar, model, lr=.001, iterations=50, KLD_weight=1):
    """ Do SGD on latent z to minimize loss for a given model, label
    mu, logvar must be proper format leaf variables
    """
    model=model.eval()
    for p in model.parameters():
        p.requires_grad=False

    BCE_list = []
    KLD_list = []
    
    optimizer = optim.Adam([mu, logvar], lr=lr)

    for i in range(iterations):
        z = model.reparameterize(mu, logvar)
        recon_x = model.decode(z, labels)
        loss, BCE, KLD = VAE_test_loss(recon_x, mu, logvar, imgs, KLD_weight)
        BCE_list.append(BCE.item())
        KLD_list.append(KLD.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return recon_x, z, loss, BCE_list, KLD_list


def gen_classify_cvae(img, labels, model, num_times, iterations, latent_size, device, KLD_weight=1):
    """Classifiy the sample as the class corresponding to minumum KLD loss found using num_times random restarts of SGD
    Img - Pytorch format
    labels - numpy array

    """
    results = {}
    for label in labels:
        results[label] = []

    for trial in range(num_times):
        for label in labels:
            tensor_label = torch.from_numpy(np.array(label)).unsqueeze(0).type(torch.LongTensor).to(device)

            rand_mu = np.random.normal(0,1, (1, latent_size))
            rand_logvar = np.random.normal(0,1, (1, latent_size))
            mu = torch.tensor(rand_mu, device=device, requires_grad=True).type(torch.cuda.FloatTensor)
            logvar = torch.tensor(rand_logvar, device=device, requires_grad=True).type(torch.cuda.FloatTensor)
            mu = Variable(mu.data, requires_grad=True)
            logvar = Variable(logvar.data, requires_grad=True)

            recon_x, z, loss, BCE_list, KLD_list = optimize_latent_cvae(img, tensor_label, mu, logvar, model, 
                                                                        lr=.001, iterations=iterations, KLD_weight=KLD_weight)
            results[label].append(loss.item())

    mins = {k:np.min(v) for (k,v) in results.items()}
    predicted_label = min(mins, key=mins.get)
    return results, predicted_label

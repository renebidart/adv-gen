import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.nn import functional as F
import torch.optim as optim


class GenerativeCVAE(nn.Module):
    """
    ??? ONLY USE BATCH SIZE OF 1 - Remove all the numpy stuff for results
    ??? Not doing classification the same way as with eq.4 https://arxiv.org/pdf/1805.09190v3.pdf
    Classify using a conditional VAE.
    
    Do gradient descent to update using each class to reconstruct the image.
    Initialized latent as random normal, use adam
    labels is a list of labels
    """
    def __init__(self, model, labels, latent_size, device):
        super(GenerativeCVAE, self).__init__()
        self.labels = labels
        self.latent_size = latent_size
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad=False
        self.device = device

    def forward(self, img, iterations=50, num_times=100, KLD_weight=1, info=False, deterministic=True): # add in the init?
        lr = .001
        results = np.tile(1000000000, (num_times, len(self.labels))) # no batch size
        results = torch.from_numpy(results).float().to(self.device)

        for trial in range(num_times):
            for label in self.labels:
                rand_mu = np.random.normal(0,1, (1, self.latent_size))
                rand_logvar = np.random.normal(0,1, (1, self.latent_size))
                mu = torch.from_numpy(rand_mu).float().to(self.device)
                logvar = torch.from_numpy(rand_logvar).float().to(self.device)

                tensor_label = torch.from_numpy(np.array(label)).unsqueeze(0).type(torch.LongTensor).to(self.device)
                optimizer = optim.Adam([mu, logvar], lr=lr)

                for i in range(iterations):
                    z = self.model.reparameterize(mu, logvar, deterministic=deterministic)
                    recon_x = self.model.decode(z, tensor_label)
                    output = (recon_x, mu, logvar) # put it in format for the loss
                    loss = self.model.loss(output, img, KLD_weight=KLD_weight)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                results[trial, label] = -1 * loss.item() # this is not the correct loss, but will work

        results = torch.min(results, 0)[0]
        predictions = F.softmax(results, dim=0)
        predictions = predictions.unsqueeze(0) # add the batch dim
        return predictions


class GenerativeVAE(nn.Module):
    """ Classify using a set of VAEs

    ??? ONLY USE BATCH SIZE OF 1 - Remove all the numpy stuff for results
    ??? Not doing classification the same way as with eq.4 https://arxiv.org/pdf/1805.09190v3.pdf
    Do gradient descent to update using each class to reconstruct the image.
    Initialized latent as random normal, use adam
    labels is a list of each label
    """
    def __init__(self, model_dict, labels, latent_size, device):
        super(GenerativeVAE, self).__init__()
        self.labels = labels
        self.model_dict = model_dict
        self.latent_size = latent_size
        self.device = device

    def forward(self, img, iterations=50, num_times=100, KLD_weight=1, info=False, deterministic=True): # add in the init?
        lr = .001
        results = np.tile(-100000, (num_times, len(self.model_dict.keys()))) # no batch size
        results = torch.from_numpy(results).float().to(self.device)

        for label, model in self.model_dict.items():
            model = model.eval()
            for p in model.parameters():
                p.requires_grad=False

        for trial in range(num_times):
            for label, model in self.model_dict.items():
                rand_mu = np.random.normal(0,1, (1, self.latent_size))
                rand_logvar = np.random.normal(0,1, (1, self.latent_size))
                mu = torch.from_numpy(rand_mu).float().to(self.device)
                logvar = torch.from_numpy(rand_logvar).float().to(self.device)

                tensor_label = torch.from_numpy(np.array(label)).unsqueeze(0).type(torch.LongTensor).to(self.device)
                optimizer = optim.Adam([mu, logvar], lr=lr)

                for i in range(iterations):
                    z = model.reparameterize(mu, logvar, deterministic=deterministic)
                    recon_x = model.decode(z)
                    output = (recon_x, mu, logvar) # put it in format for the loss
                    loss = model.loss(output, img, KLD_weight=KLD_weight)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                results[trial, label] = -1 * loss.item() # this is not the correct loss, but will work

        results = torch.min(results, 0)[0]
        predictions = F.softmax(results, dim=0)
        predictions = predictions.unsqueeze(0) # add the batch dim
        
        if info:
            predictions, results
        return predictions
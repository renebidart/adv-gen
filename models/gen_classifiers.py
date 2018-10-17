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

    def forward(self, img, iterations=50, num_times=100, KLD_weight=1, info=False): # add in the init?
        lr = .001
#         results = {}
#         for label in self.labels:
#             results[label] = []
#         results = np.repeat(1000000000, num_times*len(self.labels), (num_times, len(self.labels))) # no batch size
        results = np.tile(1000000000, (num_times, len(self.labels))) # no batch size

        results = torch.from_numpy(results).float().to(self.device)

        for trial in range(num_times):
            rand_mu = np.random.normal(0,1, (1, self.latent_size))
            rand_logvar = np.random.normal(0,1, (1, self.latent_size))
            mu = torch.from_numpy(rand_mu).float().to(self.device)
            logvar = torch.from_numpy(rand_logvar).float().to(self.device)

            for label in self.labels:                
                tensor_label = torch.from_numpy(np.array(label)).unsqueeze(0).type(torch.LongTensor).to(self.device)
                optimizer = optim.Adam([mu, logvar], lr=lr)

                for i in range(iterations):
                    z = self.model.reparameterize(mu, logvar, deterministic=True)
                    recon_x = self.model.decode(z, tensor_label)
                    output = (recon_x, mu, logvar) # put it in format for the loss
                    loss = self.model.loss(output, img, KLD_weight=KLD_weight, single_batch=False) # assume single img
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                results[trial, label] = loss.item()
            # print(f'results:, {results}')

#                 results.scatter_(dim=0, index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), src=loss.item())
#                 results[trial, label.item()].append(loss.item())
#         mins = {k:np.min(v) for (k,v) in results.items()}
#         predicted_label = min(mins, key=mins.get)
#         predictions = torch.exp(results)/torch.sum(torch.exp(results))
        print(f'results.size(), {results.size()}')
        results = torch.min(results, 0)[0]
        print(f'results(min):, {results}')
        print(f'results.size(), {results.size()}')
        predictions = F.softmax(results, dim=0)
        print(f'predictions:, {predictions}')
        predictions = predictions.unsqueeze(0) # add the batch dim
        
        if info:
            predictions, results
        return predictions

# class GenerativeCVAE(nn.Module):
#     """
#     ??? ONLY USE BATCH SIZE OF 1 - Remove all the numpy stuff for results
#     Classify using a conditional VAE.
    
#     Do gradient descent to update using each class to reconstruct the image.
#     Initialized latent as random normal, use adam
#     labels is a pytorch tensor of labels
#     """
#     def __init__(self, model, labels, latent_size, device):
#         super(GenerativeCVAE, self).__init__()
#         self.labels = labels
#         self.latent_size = latent_size
#         self.model = model.eval()
#         for p in self.model.parameters():
#             p.requires_grad=False
#         self.device = device

#     def forward(self, img, iterations=50, num_times=100, KLD_weight=1, info=False): # add in the init?
#         lr = .001
#         results = {}
#         for label in self.labels:
#             results[label] = []

#         for trial in range(num_times):
#             rand_mu = np.random.normal(0,1, (1, self.latent_size))
#             rand_logvar = np.random.normal(0,1, (1, self.latent_size))
#             mu = torch.from_numpy(rand_mu).float().to(self.device)
#             logvar = torch.from_numpy(rand_logvar).float().to(self.device)

#             for label in self.labels:                
#                 tensor_label = torch.from_numpy(np.array(label)).unsqueeze(0).type(torch.LongTensor).to(self.device)
#                 optimizer = optim.Adam([mu, logvar], lr=lr)

#                 for i in range(iterations):
#                     z = self.model.reparameterize(mu, logvar, deterministic=False)
#                     recon_x = self.model.decode(z, tensor_label)
#                     output = (recon_x, mu, logvar) # put it in format for the loss
#                     loss = self.model.loss(output, img, KLD_weight=KLD_weight, fooldumb=True)
                    
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                 results[label].append(loss.item())
    
#         mins = {k:np.min(v) for (k,v) in results.items()}
#         predicted_label = min(mins, key=mins.get)
#         predicted_label = torch.from_numpy(np.array(predicted_label)).to(self.device).unsqueeze(0).unsqueeze(0)
#         print(predicted_label.size())

#         if info:
#             results, predicted_label
#         return predicted_label

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

    def forward(self, img, iterations=50, num_times=100,KLD_weight=1, info=False): # add in the init?
        lr = .001
        results = np.tile(1000000000, (num_times, len(self.model_dict.keys()))) # no batch size
        results = torch.from_numpy(results).float().to(self.device)

        for label, model in self.model_dict.items():
            model = model.eval()
            for p in model.parameters():
                p.requires_grad=False

        for trial in range(num_times):
            rand_mu = np.random.normal(0,1, (1, self.latent_size))
            rand_logvar = np.random.normal(0,1, (1, self.latent_size))
            mu = torch.from_numpy(rand_mu).float().to(self.device)
            logvar = torch.from_numpy(rand_logvar).float().to(self.device)
            
            for label, model in self.model_dict.items():
                tensor_label = torch.from_numpy(np.array(label)).unsqueeze(0).type(torch.LongTensor).to(self.device)
                optimizer = optim.Adam([mu, logvar], lr=lr)

                for i in range(iterations):
                    z = model.reparameterize(mu, logvar)
                    recon_x = model.decode(z)
                    output = (recon_x, mu, logvar) # put it in format for the loss
                    loss = model.loss(output, img, KLD_weight=KLD_weight)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                results[trial, label] = loss.item()

        results = torch.min(results, 0)[0]
        predictions = F.softmax(results, dim=0)
        predictions = predictions.unsqueeze(0) # add the batch dim

        if info:
            predictions, results
        return predictions
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class VAE_ABS(nn.Module):
    """VAE based off https://arxiv.org/pdf/1805.09190v3.pdf
    ??? SHould we use global avg pooling and a 1x1 conv to get mu, sigma? Or even no 1x1, just normal conv.

    should the first fc in deconv be making the output batch*8*7*7???
    """
    def __init__(self, latent_size=8, img_size=32):
        super(VAE_ABS, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.linear_size = 7*7*16

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_conv4 = nn.Conv2d(64, 2*8, kernel_size=5, stride=1, padding=2, bias=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)

        self.dec_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, padding=2,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv4 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=True)

        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_bn3 = nn.BatchNorm2d(16)

        self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size, self.linear_size)

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, deterministic)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def encode(self, x):
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.elu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view((-1, 16, int(self.img_size/4), int(self.img_size/4)))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.elu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_conv4(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

    def loss(self, output, x, KLD_weight=1, info=False):
        recon_x, mu, logvar = output
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        loss = Variable(BCE+KLD_weight*KLD, requires_grad=True)
        if info:
            return loss, BCE, KLD
        return loss



class VAE_general(nn.Module):
    """VAE
    Add one-hot encoded class label as input to the last FC layer in encoder
    """
    def __init__(self, latent_size=32, img_size=32, layer_sizes=[3, 32, 64, 128]):
        super(VAE_general, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.layer_sizes = layer_sizes
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.final_im_size = int(self.img_size/(2**(len(self.layer_sizes)-1)))
        self.linear_size = int(self.final_im_size**2*self.layer_sizes[-1])

        # Encoder
        self.encoder = nn.ModuleList()
        for i, layer_size in enumerate(layer_sizes[:-1]):
            self.encoder.append(nn.Conv2d(layer_size, layer_sizes[i+1], kernel_size=4, stride=2, padding=1, bias=True))
            self.encoder.append(nn.BatchNorm2d(layer_sizes[i+1]))
            self.encoder.append(self.leakyrelu)
        self.encoder = nn.Sequential(*self.encoder)

        # FC
        self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size, self.latent_size)
        self.fc1 = nn.Linear(self.latent_size, self.linear_size)

        # Decoder
        self.decoder = nn.ModuleList()
        for i, layer_size in enumerate(self.layer_sizes[::-1][:-1]):
            self.decoder.append(Interpolate(scale_factor=2, mode='nearest'))
            self.decoder.append(nn.ReplicationPad2d(1))
            self.decoder.append(nn.Conv2d(layer_size, self.layer_sizes[::-1][i+1], kernel_size=3, stride=1, bias=True))
            self.decoder.append(nn.BatchNorm2d(self.layer_sizes[::-1][i+1]))
            self.decoder.append(self.leakyrelu)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x, training=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, x): # this is messy, really should put fc1 and decoder together into one thing.
        x = self.fc1(x)
        x = x.view((-1, self.layer_sizes[-1], self.final_im_size, self.final_im_size))
        x = self.decoder(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, logvar, inference=False):
        if inference:
            return mu
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

    def loss(self, output, inputs):
        x = inputs
        recon_x, mu, logsigma = output
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD

# what the fuck
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=None)
        return x
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class FEAT_VAE_MNIST(nn.Module):
    """VAE based off https://arxiv.org/pdf/1805.09190v3.pdf
    ??? SHould we use global avg pooling and a 1x1 conv to get mu, sigma? Or even no 1x1, just normal conv.

    should the first fc in deconv be making the output batch*8*7*7???
    """
    def __init__(self, encoding_model, latent_size=6, num_features=None):
        super(FEAT_VAE_MNIST, self).__init__()
        self.encoding_model = encoding_model.eval()
        
        self.num_features = num_features
        self.latent_size = latent_size
        self.linear_size = 4*4*16

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(self.num_features, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_bn2 = nn.BatchNorm2d(16)

        self.dec_conv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1,  output_padding=0, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(16, self.num_features, kernel_size=3, padding=1,  output_padding=0, bias=False)
        self.dec_bn1 = nn.BatchNorm2d(16)
        self.dec_bn2 = nn.BatchNorm2d(16)

        self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size, self.linear_size)

        for p in self.encoding_model.parameters():
            p.requires_grad=False


    def forward(self, x, label, deterministic=False):  # label actually doesn't do anything, just for consistency with other cvae?
        x = self.encoding_model.encode_feat(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, deterministic)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def encode(self, x):
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view((-1, 16, 4, 4))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

    def loss(self, output, x, KLD_weight=1, info=False):
        """Compute the loss between the encoded features, and the generated features, not the image"""
        x = self.encoding_model.encode_feat(x)
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

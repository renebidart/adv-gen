import torch
import torch.nn as nn
from torch.nn import functional as F


class CVAE(nn.Module):
    """Conditional VAE
    use nn.ReplicationPad2d?
    Add one-hot encoded class label as input to the last FC layer in encoder
    """
    def __init__(self, num_labels=1, latent_size=32, img_size=32, layer_sizes=[3, 32, 64, 128]):
        super(CVAE, self).__init__()
        self.num_labels = num_labels
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
        self.fc_mu = nn.Linear(self.linear_size+self.num_labels, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size+self.num_labels, self.latent_size)
        self.fc1 = nn.Linear(self.latent_size+self.num_labels, self.linear_size)

        # Decoder
        self.decoder = nn.ModuleList()
        for i, layer_size in enumerate(self.layer_sizes[::-1][:-1]):
            self.decoder.append(nn.UpsamplingNearest2d(scale_factor=2))
            self.decoder.append(nn.ReplicationPad2d(1))
            self.decoder.append(nn.Conv2d(layer_size, self.layer_sizes[::-1][i+1], kernel_size=3, stride=1, bias=True))
            self.decoder.append(nn.BatchNorm2d(self.layer_sizes[::-1][i+1]))
            self.decoder.append(self.leakyrelu)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x, c, training=True):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


    def encode(self, x, c):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, self.to_one_hot(c).type(x.type())), dim=1)
        return self.fc_mu(x), self.fc_logvar(x) 

    def decode(self, x, c):
        x = torch.cat((x, self.to_one_hot(c).type(x.type())), dim=1)
        x = self.fc1(x)
        x = x.view((-1, self.layer_sizes[-1], self.final_im_size, self.final_im_size))
        x = self.decoder(x)
        return F.sigmoid(x)

    def reparameterize(self, mu, logvar):
        if self.training: # return random normal with the correct mu, sigma
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss(self, output, inputs):
        x = inputs
        recon_x, mu, logsigma = output
        BCE = F.mse_loss(recon_x, x, size_average=False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD

    def to_one_hot(self, y):
        y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size()[0], self.num_labels).type(y.type())#torch.cuda.FloatTensor) #y.type()
        y_onehot.scatter_(1, y, 1)
        return y_onehot
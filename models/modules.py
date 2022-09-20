from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class XrayBaseVAE:
    def __init__(self):
        self.bce_dim = 0
        self.use_mse = True

    def encode(self, x):
        h = self.encoder(x)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        return x

    def loss_unsupervised(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        #print(recon_x.shape, x.shape)
        if self.use_mse:
            BCE = F.mse_loss(recon_x.view(-1, self.bce_dim), x.view(-1, self.bce_dim), reduction='sum')
        else:
            BCE = F.binary_cross_entropy(recon_x.view(-1, self.bce_dim), x.view(-1, self.bce_dim), reduction='sum')

        KLD = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logvar) - 1 - logvar)
        loss = BCE + self.dist_weight * KLD

        return loss

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size()[0], -1)

class UnFlatten(nn.Module):

    def __init__(self, shape):
        super(UnFlatten, self).__init__()

        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), self.shape[0], self.shape[1], self.shape[2])


class VAE_Xray(nn.Module, XrayBaseVAE):
    def __init__(self, latent_size=50, dist_weight=1):
        super(VAE_Xray, self).__init__()

        self.latent_size = latent_size
        self.bce_dim = 244 * 244
        self.dist_weight = dist_weight
        self.use_mse = False

        self.encoder = xray_encoder()
        self.decoder = xray_decoder(latent_dim=latent_size)
        self.fc21 = nn.Linear(256, self.latent_size)
        self.fc22 = nn.Linear(256, self.latent_size)

    def forward(self, x):
        return XrayBaseVAE.forward(self, x)

def xray_encoder(flatten_dim=(128, 58, 58)):
    encoder = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=1),
        nn.BatchNorm2d(16),

        nn.ReLU(),

        nn.Conv2d(16, 32, 3, stride=2),
        nn.BatchNorm2d(32),

        nn.ReLU(),

        nn.Conv2d(32, 64, 3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 128, 3, stride=2),
        nn.BatchNorm2d(128),

        nn.ReLU(),

        Flatten(),
        nn.Linear(flatten_dim[0] * flatten_dim[1] * flatten_dim[2], 256),
        nn.BatchNorm1d(256),
        nn.ReLU()

    )
    return encoder

def xray_decoder(latent_dim=50, unflatten_dim=(128, 58, 58)):
    decoder = nn.Sequential(

        nn.Linear(latent_dim, 256),
        nn.ReLU(),

        nn.Linear(256, unflatten_dim[0] * unflatten_dim[1] * unflatten_dim[2]),
        nn.ReLU(),

        UnFlatten(unflatten_dim),

        nn.ConvTranspose2d(128, 64, 5, stride=2),
        nn.ReLU(),

        nn.ConvTranspose2d(64, 32, 4, stride=2),
        nn.ReLU(),

        nn.ConvTranspose2d(32, 16, 4, stride=1),
        nn.ReLU(),

        nn.ConvTranspose2d(16, 1, 2, stride=1),
        nn.Sigmoid(),
    )
    return decoder
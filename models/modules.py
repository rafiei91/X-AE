from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.nn.init import orthogonal_

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


class XrayBaseRVAE(XrayBaseVAE):
    def __init__(self):
        self.bce_dim = 0
        self.dist_weight_2 = 1
        self.use_mse = False

    def loss_unsupervised(self, recon_x, x, mu, logvar):

        if self.use_mse:
            BCE = F.mse_loss(recon_x.view(-1, self.bce_dim), x.view(-1, self.bce_dim), reduction='sum')
        else:
            BCE = F.binary_cross_entropy(recon_x.view(-1, self.bce_dim), x.view(-1, self.bce_dim), reduction='sum')

        # Get the nearest center
        W = self.centers.weight
        distances = pairwise_squared_distances(mu, W)
        idx = torch.argmin(distances, 1)
        mus = self.centers.weight[idx, :]

        KLD = 0.5 * torch.sum(torch.pow(mu - mus, 2) + torch.exp(logvar) - 1 - logvar)

        return BCE + self.dist_weight_2 * KLD

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_supervised(self, recon_x, x, mu, logvar, labels):
        # Reconstruction loss
        if self.use_mse:
            BCE = F.mse_loss(recon_x.view(-1, self.bce_dim), x.view(-1, self.bce_dim), reduction='sum')
        else:
            BCE = F.binary_cross_entropy(recon_x.view(-1, self.bce_dim), x.view(-1, self.bce_dim), reduction='sum')

        mus = self.centers.weight[labels, :]
        KLD = 0.5 * torch.sum(torch.pow(mu - mus, 2) + torch.exp(logvar) - 1 - logvar)

        return BCE + self.dist_weight * KLD

    def loss_gaussians(self):
        W = self.centers.weight
        D = pairwise_squared_distances(W, W)

        mask = 1 - torch.eye(D.size(0)).cuda()
        loss = ((D - self.gaussian_target_distance) ** 2) * mask

        return torch.sum(loss) / self.gaussian_target_distance


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
    def __init__(self, latent_size=50, dist_weight=1, use_mse = False):
        super(VAE_Xray, self).__init__()

        self.latent_size = latent_size
        self.bce_dim = 244 * 244
        self.dist_weight = dist_weight
        self.use_mse = use_mse

        self.encoder = xray_encoder()
        self.decoder = xray_decoder(latent_dim=latent_size)
        self.fc21 = nn.Linear(256, self.latent_size)
        self.fc22 = nn.Linear(256, self.latent_size)

    def forward(self, x):
        return XrayBaseVAE.forward(self, x)


class RVAE_Xray(nn.Module, XrayBaseRVAE):
    def __init__(self, latent_size=6, n_modalities=10, gaussian_target_distance=3, dist_weight=1, use_mse = False):
        super(RVAE_Xray, self).__init__()

        self.latent_size = latent_size
        self.gaussian_target_distance = (gaussian_target_distance) * latent_size
        self.bce_dim = 244 * 244
        self.dist_weight = dist_weight
        self.dist_weight_2 = 1
        self.use_mse = use_mse

        self.encoder = xray_encoder()
        self.decoder = xray_decoder(latent_dim=latent_size)
        self.fc21 = nn.Linear(256, self.latent_size)
        self.fc22 = nn.Linear(256, self.latent_size)

        self.centers = nn.Linear(self.latent_size, n_modalities, bias=False)
        self.centers.weight.data = orthogonal_(self.centers.weight.data, gain=gaussian_target_distance)

    def forward(self, x):
        return XrayBaseRVAE.forward(self, x)

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

def pairwise_squared_distances(X, Y):
    X_squared = torch.sum(X ** 2, 1, keepdim=True)
    Y_squared = torch.sum(Y ** 2, 1, keepdim=True).transpose(0, 1)

    XY = torch.mm(X, Y.transpose(0, 1))

    dists = X_squared + Y_squared - 2 * XY
    dists = torch.clamp(dists, 1e-7, 1e6)
    # dists = torch.sqrt(dists)

    return dists
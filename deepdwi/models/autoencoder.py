"""
This module defines the autoencoder (AE) models.

References:


Author:
    Soundarya Soundarresan <soundarya.soundarresan@fau.de>
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch

import torch.nn as nn

from torch.nn import functional as F

from typing import List


# %%
class DAE(nn.Module):
    """
    Denoising AutoEncoder
    """
    def __init__(self,
                 input_features: int = 81,
                 latent_features: int = 15,
                 depth: int = 4,
                 encoder_features: List[int] = None,
                 use_conv: bool = False):

        super(DAE, self).__init__()

        encoder_module = []
        decoder_module = []

        if encoder_features is None:

            encoder_features = torch.linspace(start=input_features, end=latent_features, steps=depth+1).type(torch.int64)

        else:

            encoder_features = torch.tensor(encoder_features)

        #     assert(depth == len(encoder_features))


        decoder_features = torch.flip(encoder_features, dims=(0, ))



        for d in range(depth):
            encoder_module.append(nn.Linear(encoder_features[d], encoder_features[d+1]))
            encoder_module.append(nn.ReLU(True))

            decoder_module.append(nn.Linear(decoder_features[d], decoder_features[d+1]))
            if d < (depth - 1):
                decoder_module.append(nn.ReLU(True))
            else:
                decoder_module.append(nn.Sigmoid())

        self.encoder_seq = nn.Sequential(*encoder_module)
        self.decoder_seq = nn.Sequential(*decoder_module)

    def encode(self, x):
        return self.encoder_seq(x)

    def decode(self, x):
        return self.decoder_seq(x)

    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)

        return output


# %%
class VAE(nn.Module):
    """
    Variational AutoEncoder
    """
    def __init__(self,
                 input_features=81,
                 latent_features=15,
                 depth=4):

        super(VAE, self).__init__()

        encoder_module = []
        decoder_module = []

        encoder_features = torch.linspace(start=input_features,
                                          end=latent_features,
                                          steps=depth+1).type(torch.int64)
        decoder_features = torch.flip(encoder_features, dims=(0, ))

        # encoder
        for d in range(depth - 1):
            encoder_module.append(nn.Linear(encoder_features[d], encoder_features[d+1]))
            encoder_module.append(nn.ReLU(True))

        self.encoder_seq = nn.Sequential(*encoder_module)

        # latent layer
        self.fc1 = nn.Linear(encoder_features[depth-1], encoder_features[depth])
        self.fc2 = nn.Linear(encoder_features[depth-1], encoder_features[depth])

        # decoder
        for d in range(depth):
            decoder_module.append(nn.Linear(decoder_features[d], decoder_features[d+1]))
            if d < (depth - 1):
                decoder_module.append(nn.ReLU(True))
            else: # last layer
                decoder_module.append(nn.Sigmoid())

        self.decoder_seq = nn.Sequential(*decoder_module)

    def encode(self, x):
        features = self.encoder_seq(x)
        return self.fc1(features), self.fc2(features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_seq(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function_mse(recon, orig):
    loss = F.mse_loss(recon, orig)

    if loss.isnan():
        return torch.tensor([1E-6])
    else:
        return loss

def loss_function_l1(recon, orig):
    loss = F.l1_loss(recon, orig)

    if loss.isnan():
        return torch.tensor([1E-6])
    else:
        return loss


def loss_function_kld( mu=None, logvar=None):
    """
    Reconstruction + KL divergence losses summed over all elements and batch

    Reference:
        * Kingma DP, Welling M.
          Auto-encoding Variational Bayes. ICLR (2014).
    Split up KLD and reconstruction loss
    Assume pdfs to be Gaussian to use analytical formula
    """

    if mu is None:
        mu = torch.tensor([0])

    if logvar is None:
        logvar = torch.tensor([0])

    std = torch.exp(0.5 * logvar)

    KLD = -0.5 * torch.mean(1 + std.log() - mu.pow(2) - std)

    return KLD

# model.py

import torch
import torch.nn as nn
from Utils.modules import *


class HierarchicalVAE(nn.Module):
    def __init__(self, config):
        super(HierarchicalVAE, self).__init__()
        self.config = config
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()

    def _make_encoder(self):
        encoder_layers = []
        in_channels = self.config['in_channels']
        latent_dim = self.config['latent_dim']

        for i in range(self.config['num_encoder_blocks']):
            out_channels = self.config['encoder_channels'][i]
            stride = 2 if i > 0 else 1
            encoder_layers.append(EncoderBlock(in_channels, out_channels, stride))
            in_channels = out_channels

        self.encoder_output_size = in_channels * 4 * 4  # Assuming input size is (3, 64, 64)

        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(self.encoder_output_size, latent_dim * 2))  # *2 for mean and std
        return nn.Sequential(*encoder_layers)

    def _make_decoder(self):
        decoder_layers = []
        latent_dim = self.config['latent_dim']
        out_channels = self.config['encoder_channels'][-1]  # Assuming symmetric architecture

        decoder_layers.append(nn.Linear(latent_dim, self.encoder_output_size))
        decoder_layers.append(nn.Unflatten(1, (out_channels, 4, 4)))

        for i in range(self.config['num_decoder_blocks']):
            in_channels = out_channels
            out_channels = self.config['decoder_channels'][i]
            stride = 2 if i < self.config['num_decoder_blocks'] - 1 else 1
            decoder_layers.append(DecoderBlock(in_channels, out_channels, stride))

        decoder_layers.append(
            nn.ConvTranspose2d(out_channels, self.config['out_channels'], kernel_size=4, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())  # Assuming output is in [0, 1]
        return nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :self.config['latent_dim']]
        logvar = mu_logvar[:, self.config['latent_dim']:]
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

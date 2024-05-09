# model.py

import torch
import torch.nn as nn
from Utils.modules import ResidualCell


class HVAE(nn.Module):
    def __init__(self, model_config):
        super(HVAE, self).__init__()
        self.model_config = model_config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder_blocks = nn.ModuleList()
        in_channels = self.model_config.input_channels
        out_channels_list = self.model_config.encoder_channels
        for i in range(self.model_config.num_encoder_layers):
            out_channels = out_channels_list[i]
            encoder_blocks.append(ResidualCell(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        # Add a final layer to predict delta mu and delta logvar
        encoder_blocks.append(nn.Conv2d(in_channels, 2 * self.model_config.latent_dim, kernel_size=1))
        return encoder_blocks

    def _build_decoder(self):
        decoder_blocks = nn.ModuleList()
        in_channels = sum(self.model_config.encoder_channels)
        for i in range(self.model_config.num_decoder_layers):
            out_channels = self.model_config.decoder_channels[i]
            decoder_blocks.append(ResidualCell(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        return decoder_blocks

    def forward(self, x):
        # Encode
        encoder_outputs = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
        # Flatten the output for further processing
        x = torch.flatten(x, start_dim=1)
        # Split the output into delta mu and delta logvar
        delta_mu, delta_logvar = torch.split(x, self.model_config.latent_dim, dim=1)

        # Decode
        for decoder_block in self.decoder:
            x = decoder_block(x)

        return x, delta_mu, delta_logvar
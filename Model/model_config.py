# model_config.py

model_config = {
    'in_channels': 3,
    'out_channels': 3,  # Assuming RGB images
    'latent_dim': 128,  # Latent space dimension
    'num_encoder_blocks': 4,  # Number of encoder blocks
    'encoder_channels': [32, 64, 128, 256],  # Encoder block channels
    'num_decoder_blocks': 4,  # Number of decoder blocks
    'decoder_channels': [128, 64, 32, 16],  # Decoder block channels
}

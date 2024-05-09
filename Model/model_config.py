# model_config.py

class ModelConfig:
    def __init__(self):
        # Encoder parameters
        self.num_encoder_layers = 4
        self.encoder_channels = [64, 128, 256, 512]  # Number of channels in each encoder layer

        # Decoder parameters
        self.num_decoder_layers = 4
        self.decoder_channels = [256, 128, 64, 32]  # Number of channels in each decoder layer

        # Input image parameters
        self.input_channels = 3  # RGB channels
        self.image_size = 256  # Input image size (assumed to be square)
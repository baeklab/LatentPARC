import torch
import torch.nn as nn

# Convolutional AE
class Encoder(nn.Module):
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
        super().__init__()
        modules = []
        in_channels = layers[0]
        for out_channels in layers[1:]:
            modules.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )  # Keep padding=1 for same-sized convolutions
            modules.append(act_fn)
            in_channels = out_channels
        modules.append(
            nn.Conv2d(layers[-1], latent_dim, kernel_size=3, stride = 2, padding=1)
        )  # Bottleneck layer
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):    # no deconv
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
        super().__init__()

        self.in_channels = layers[-1]
        self.latent_dim = latent_dim

        modules = []
        in_channels = latent_dim #layers[-1]

        # Initial convolution layer for latent vector
        # modules.append(nn.Conv2d(latent_dim, in_channels, kernel_size=3, padding=1))

        # Iteratively create resize-convolution layers
        for out_channels in reversed(layers): #layers[:-1]
            modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))  # Resizing
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # Convolution
            modules.append(act_fn)  # Activation function
            in_channels = out_channels
            
        # modules.pop() # final activation linear
        # modules.append(nn.Sigmoid())
        
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


# Defining the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Wrappers
class ConvolutionalAutoencoder:
    def __init__(self, autoencoder, optimizer, device, save_path=None, weights_name=None):
        self.network = autoencoder.to(device)
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.weights_name = weights_name

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        return self.network.encoder(x)

    def decode(self, x):
        return self.network.decoder(x)
    
### Future TODO: replace Autoencoder and ConvolutionalAutoencoder classes with below, current is redundant
### check in train loop anywhere we reference model.network and replace those with just model since the 
### nn.Module is now the top-level object directly
# class Autoencoder(nn.Module):
#     def __init__(self, encoder, decoder, optimizer=None, device='cpu', save_path=None, weights_name=None):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.optimizer = optimizer
#         self.device = device
#         self.save_path = save_path
#         self.weights_name = weights_name
#         self.to(device)

#     def forward(self, x):
#         return self.decoder(self.encoder(x))

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, x):
#         return self.decoder(x)
    
    
class Autoencoder_separate(nn.Module):
    def __init__(self, encoder_T, encoder_P, encoder_M, decoder_T, decoder_P, decoder_M):
        super().__init__()
        self.encoderT = encoder_T
        self.encoderP = encoder_P
        self.encoderM = encoder_M
        self.decoderT = decoder_T
        self.decoderP = decoder_P
        self.decoderM = decoder_M
    
    def forward(self, x):
        z_t = self.encoderT(x[:, 0:1, :, :]) # only T channel
        z_p = self.encoderP(x[:, 1:2, :, :]) # only P channel
        z_m = self.encoderM(x[:, 2:3, :, :]) # only M channel
                
        decoded_t = self.decoderT(z_t) # decode T
        decoded_p = self.decoderP(z_p) # decode P
        decoded_m = self.decoderM(z_m) # decode M
        decoded = torch.cat((decoded_t, decoded_p, decoded_m), dim=1) # concat for output
        
        return decoded
    
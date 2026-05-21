import torch
import torch.nn as nn

# Convolutional AE
class Encoder(nn.Module): # base model
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

class EncoderWithSkips(nn.Module): # for skip con super resolution model
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
        super().__init__()

        # Build each (Conv + Act) pair as a separate block
        # so we can call them one at a time and save intermediates
        self.blocks = nn.ModuleList()
        in_channels = layers[0]
        for out_channels in layers[1:]:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                act_fn
            ))
            in_channels = out_channels

        # Bottleneck layer stays the same
        self.bottleneck = nn.Conv2d(layers[-1], latent_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        skips = []
        out = x
        for block in self.blocks:
            out = block(out)
            skips.append(out)  # save every intermediate feature map
        z = self.bottleneck(out)
        return z, skips  # latent + all skip feature maps
    
class SRHeadWithSkip(nn.Module): # for skip con super resolution model
    def __init__(self, layers, output_channels, act_fn=nn.ReLU()):
        super().__init__()

        skip_ch  = layers[1]
        out_ch   = output_channels

        # Only project if the skip is wide enough to warrant compression.
        # If skip_ch is already small, use it directly and skip the 1x1 entirely.
        proj_ch = min(skip_ch, 32)  # never expand, only compress or keep as-is
        
        if skip_ch > proj_ch:
            self.skip_proj = nn.Conv2d(skip_ch, proj_ch, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()  # no-op, pass skip straight through

        self.net = nn.Sequential(
            nn.Conv2d(out_ch + proj_ch, 64, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, blurry, skips):
        skip = self.skip_proj(skips[0])

        if skip.shape[-2:] != blurry.shape[-2:]:
            skip = nn.functional.interpolate(
                skip, size=blurry.shape[-2:],
                mode='bilinear', align_corners=False
            )

        fused = torch.cat([blurry, skip], dim=1)
        return blurry + self.net(fused)

class SRHead(nn.Module): 
    def __init__(self, output_channels, act_fn=nn.ReLU(), out_act=None):
        super().__init__()
        out_ch = output_channels
        self.out_act = out_act
        self.net = nn.Sequential(
            nn.Conv2d(out_ch, 64, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
        )
    def forward(self, blurry):
        out = blurry + self.net(blurry)
        return self.out_act(out) if self.out_act is not None else out

class SRHeadPhong(nn.Module):
    def __init__(self, output_channels, act_fn=nn.ReLU(), out_act=None):
        super().__init__()
        out_ch = output_channels
        self.out_act = out_act
        
        self.net = nn.Sequential(
            # Wide spatial context (7x7)
            nn.Conv2d(out_ch, 64, kernel_size=7, padding=3),
            act_fn,
            # Bottleneck channel compression (1x1)
            nn.Conv2d(64, 32, kernel_size=1),
            act_fn,
            # Further compression (1x1)
            nn.Conv2d(32, out_ch, kernel_size=1),
        )
        
    def forward(self, blurry):
        out = blurry + self.net(blurry)
        return self.out_act(out) if self.out_act is not None else out

class Decoder(nn.Module):    # base model, skip con super resolution model
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()): # no deconv
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
# class BasicAutoencoder(nn.Module):
#     def __init__(self, encoder, decoder, optimizer=None, device='cpu', save_path=None, weights_name=None):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x):
#         return self.decoder(self.encoder(x))

class SkipConSuperResAutoencoder(nn.Module):
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
        super().__init__()
        self.encoder = EncoderWithSkips(layers, latent_dim, act_fn)
        self.decoder = Decoder(layers, latent_dim, act_fn)
        self.sr_head  = SRHeadWithSkip(layers, output_channels=layers[0], act_fn=act_fn)

    def forward(self, x):
        z, skips  = self.encoder(x)
        blurry    = self.decoder(z)
        sharp     = self.sr_head(blurry, skips)
        return sharp, blurry  # return both for multi-term loss


class SuperResHeadAutoencoder(nn.Module): # same as SkipConSuperResAutoencoder, but with no skip con
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU(), out_act=None):
        super().__init__()
        self.encoder = Encoder(layers, latent_dim, act_fn)
        self.decoder = Decoder(layers, latent_dim, act_fn)
        self.sr_head = SRHead(output_channels=layers[0], act_fn=act_fn, out_act=out_act)

    def forward(self, x):
        z         = self.encoder(x)
        blurry    = self.decoder(z)
        sharp     = self.sr_head(blurry)
        return sharp, blurry  # return both for multi-term loss

class SuperResHeadAutoencoderPhong(nn.Module): # same as SkipConSuperResAutoencoder, but with no skip con
    def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU(), out_act=None):
        super().__init__()
        self.encoder = Encoder(layers, latent_dim, act_fn)
        self.decoder = Decoder(layers, latent_dim, act_fn)
        self.sr_head = SRHeadPhong(output_channels=layers[0], act_fn=act_fn, out_act=out_act)

    def forward(self, x):
        z         = self.encoder(x)
        blurry    = self.decoder(z)
        sharp     = self.sr_head(blurry)
        return sharp, blurry  # return both for multi-term loss
    
class AutoencoderSeparate(nn.Module):
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
    
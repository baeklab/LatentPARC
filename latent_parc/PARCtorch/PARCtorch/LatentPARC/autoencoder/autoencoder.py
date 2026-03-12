import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
# from geometry import *

# Helper Functions #
        
def save_model(model, save_path, weights_name, epochs):
    """ Save model weights to file. """
    if save_path:
        save_file = f"{save_path}/{weights_name}_{epochs}.pth"
        torch.save(model.state_dict(), save_file)
        print(f"Model weights saved to {save_file}")

def save_log(log_dict, save_path, weights_name, epochs):
    """ Save training logs as JSON. """
    if save_path:
        log_file = f"{save_path}/{weights_name}_{epochs}.json"
        with open(log_file, 'w') as f:
            json.dump(log_dict, f)
            
def add_random_noise(images, min_val=0.0, max_val=0.1):
    """
    Add random (uniform) noise to the images.

    Parameters:
        images: Tensor of input images.
        min_val: Minimum value of the noise.
        max_val: Maximum value of the noise.

    Returns:
        Noisy images.
    """
    noise = torch.rand_like(images) * (max_val - min_val) + min_val
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)  # Keep pixel values in [0, 1]

class LpLoss(torch.nn.Module):
    def __init__(self, p=10):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, input, target):
        # Compute element-wise absolute difference
        diff = torch.abs(input - target)
        # Raise the differences to the power of p, sum them, and raise to the power of 1/p
        return (torch.sum(diff ** self.p) ** (1 / self.p))

    
# Define Modules #

class MLPEncoder(nn.Module):
    def __init__(self, layers, latent_dim, act_fn=nn.ReLU()):
        super().__init__()
        modules = []
        in_dim = layers[0]
        for dim in layers[1:]:
            modules.append(nn.Linear(in_dim, dim))
            modules.append(act_fn)
            in_dim = dim
        modules.append(nn.Linear(in_dim, latent_dim))  # Bottleneck layer
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        # Flatten input except batch dimension
        x = x.view(x.size(0), -1)
        return self.net(x)

    
class MLPDecoder(nn.Module):
    def __init__(self, layers, latent_dim, output_shape=(3, 128, 256), act_fn=nn.ReLU()):
        super().__init__()
        self.output_shape = output_shape  
        modules = []
        in_dim = latent_dim
        for dim in reversed(layers):
            modules.append(nn.Linear(in_dim, dim))
            modules.append(act_fn)
            in_dim = dim
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        x = self.net(x)
        batch_size = x.size(0)
        return x.view(batch_size, *self.output_shape)
    

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

# ### EXTRA CONV LAYER

# class Encoder(nn.Module):
#     def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
#         super().__init__()
#         modules = []
#         in_channels = layers[0]
        
#         for out_channels in layers[1:]:
#             modules.append(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
#             )
#             modules.append(act_fn)
#             modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))  # Extra conv layer
#             modules.append(act_fn)
#             in_channels = out_channels
        
#         modules.append(nn.Conv2d(layers[-1], latent_dim, kernel_size=3, stride=2, padding=1))  # Bottleneck layer
#         self.net = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.net(x)


# class Decoder(nn.Module):  
#     def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
#         super().__init__()

#         self.in_channels = layers[-1]
#         self.latent_dim = latent_dim

#         modules = []
#         in_channels = latent_dim  

#         for out_channels in reversed(layers):  
#             modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#             modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             modules.append(act_fn)
#             modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))  # Extra conv layer
#             modules.append(act_fn)
#             in_channels = out_channels
#         modules.pop() # final activation linear

#         self.conv = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.conv(x)

# RES CONNECTIONS

# class ResidualBlock(nn.Module):
#     def __init__(self, channels, act_fn=nn.ReLU()):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.act_fn = act_fn

#     def forward(self, x):
#         residual = x  # Shortcut connection
#         out = self.conv1(x)
#         out = self.act_fn(out)
#         out = self.conv2(out)
#         return self.act_fn(out + residual)  # Element-wise addition (skip connection)

# class Encoder(nn.Module):
#     def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
#         super().__init__()
#         modules = []
#         in_channels = layers[0]
        
#         for out_channels in layers[1:]:
#             modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
#             modules.append(act_fn)
#             modules.append(ResidualBlock(out_channels, act_fn))  # Residual Block
#             in_channels = out_channels

#         # Bottleneck layer
#         modules.append(nn.Conv2d(layers[-1], latent_dim, kernel_size=3, stride=2, padding=1))
#         self.net = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.net(x)

# class Decoder(nn.Module):
#     def __init__(self, layers, latent_dim=16, act_fn=nn.ReLU()):
#         super().__init__()
#         self.in_channels = layers[-1]
#         self.latent_dim = latent_dim

#         modules = []
#         in_channels = latent_dim  

#         for out_channels in reversed(layers):
#             modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#             modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
#             modules.append(act_fn)
#             modules.append(ResidualBlock(out_channels, act_fn))  # Residual Block
#             in_channels = out_channels

#         self.conv = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.conv(x)



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
    
    
# class IRCAE(nn.Module):
#     def __init__(self, encoder, decoder, iso_reg=1.0):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.iso_reg = iso_reg

#     def forward(self, x):
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         return x_hat, z  # return latent code too

# #     def compute_loss(self, x, x_hat, loss_function):
# #         recon_loss = loss_function(x_hat, x)

# #         iso_loss = relaxed_distortion_measure(self.encoder, x, eta=0.2)
# #         total_loss = recon_loss + self.iso_reg * iso_loss

# #         return total_loss, recon_loss, iso_loss
    
#     def compute_loss(self, z, x, x_hat, loss_function):
#         recon_loss = loss_function(x_hat, x)

#         iso_loss = relaxed_distortion_measure(self.decoder, z, eta=0.2)
#         total_loss = recon_loss + self.iso_reg * iso_loss

#         return total_loss, recon_loss, iso_loss
    

def train_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
                      device, epochs=10, image_size=(64, 64), n_channels=3, 
                      scheduler=None, noise_fn=None, initial_max_noise=0.16, 
                      n_reduce_factor=0.5, reduce_on=1000, save_path=None, weights_name=None):
    """ Train an autoencoder with optional noise injection. """

    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': []}
    
    model.to(device)

    max_noise = initial_max_noise  # Initial noise level

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reduce noise periodically
        if (epoch + 1) % reduce_on == 0:
            max_noise *= n_reduce_factor

        # --- Training ---
        model.train()
        train_losses = []
        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            images = images[0][:, 0:n_channels, ...].to(device)
            
            # Apply noise if function is provided
            noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

            # Forward pass
            output = model(noisy_images)
            loss = loss_function(output, images.view(-1, n_channels, *image_size))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        log_dict['training_loss_per_epoch'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_images = val_images[0][:, 0:n_channels, ...].to(device)
                
                # val_images = F.interpolate(val_images, size=(image_size[0], image_size[1]), mode='bilinear', align_corners=False) #!!! for MLP

                # Forward pass
                output = model(val_images)
                val_loss = loss_function(output, val_images.view(-1, n_channels, *image_size))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

    # Save model and logs
    save_model(model, save_path, weights_name, epochs)
    save_log(log_dict, save_path, weights_name, epochs)

    return log_dict

# def train_IR_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
#                       device, epochs=10, image_size=(64, 64), n_channels=3, 
#                       scheduler=None, noise_fn=None, initial_max_noise=0.16, 
#                       n_reduce_factor=0.5, reduce_on=1000, save_path=None, weights_name=None):
#     """ Train an autoencoder with optional noise injection. """

#     log_dict = {'training_loss_per_epoch': [], 
#                 'recon_loss_per_epoch': [],
#                 'iso_loss_per_epoch': [],
#                 'validation_loss_per_epoch': []}
    
#     model.to(device)

#     max_noise = initial_max_noise  # Initial noise level

#     for epoch in range(epochs):
#         print(f"\nEpoch {epoch + 1}/{epochs}")

#         # Reduce noise periodically
#         if (epoch + 1) % reduce_on == 0:
#             max_noise *= n_reduce_factor

#         # --- Training ---
#         model.train()
#         train_losses, recon_losses, iso_losses = [], [], []
#         for images in tqdm(train_loader, desc="Training"):
#             optimizer.zero_grad()
#             images = images[0][:, 0:n_channels, ...].to(device)
            
#             # Apply noise if function is provided
#             noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images
            
#             # Forward pass
#             x_hat, z = model(noisy_images)
#             loss, recon_loss, iso_loss = model.compute_loss(
#                 z, images.view(-1, n_channels, *image_size), x_hat, loss_function
#             )
#             loss.backward()
#             optimizer.step()
#             train_losses.append(loss.item())
#             recon_losses.append(recon_loss.item())
#             iso_losses.append(iso_loss.item())


#         avg_train_loss = np.mean(train_losses)
#         avg_recon_loss = np.mean(recon_losses)
#         avg_iso_loss = np.mean(iso_losses)
        
#         log_dict['training_loss_per_epoch'].append(avg_train_loss)
#         log_dict['recon_loss_per_epoch'].append(avg_recon_loss)
#         log_dict['iso_loss_per_epoch'].append(avg_iso_loss)

#         # --- Validation ---
#         model.eval()
#         val_losses = []
#         with torch.no_grad():
#             for val_images in tqdm(val_loader, desc="Validating"):
#                 val_images = val_images[0][:, 0:n_channels, ...].to(device)

#                 # Forward pass
#                 x_hat, z = model(val_images)
#                 loss, _, _ = model.compute_loss(
#                     z, val_images.view(-1, n_channels, *image_size), x_hat, loss_function
#                 )
#                 val_losses.append(loss.item())

#         avg_val_loss = np.mean(val_losses)
#         log_dict['validation_loss_per_epoch'].append(avg_val_loss)

#         print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

#         if scheduler:
#             scheduler.step(avg_val_loss)

#     # Save model and logs
#     save_model(model, save_path, weights_name, epochs)
#     save_log(log_dict, save_path, weights_name, epochs)

#     return log_dict


def train_individual_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
                      device, epochs=10, image_size=(64, 64), channel_index=0, 
                      scheduler=None, noise_fn=None, initial_max_noise=0.16, 
                      n_reduce_factor=0.8, reduce_on=1000, save_path=None, weights_name=None):
    
    """ Train an autoencoder on just one channel at a time with optional noise injection. """

    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': []}
    
    model.to(device)

    max_noise = initial_max_noise  # Initial noise level

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reduce noise periodically
        if (epoch + 1) % reduce_on == 0:
            max_noise *= n_reduce_factor

        # --- Training ---
        model.train()
        train_losses = []
        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            images = images[0][:, channel_index:channel_index+1, ...].to(device)
            
            # Apply noise if function is provided
            noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

            # Forward pass
            output = model(noisy_images)
            loss = loss_function(output, images.view(-1, 1, *image_size))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        log_dict['training_loss_per_epoch'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_images = val_images[0][:, channel_index:channel_index+1, ...].to(device)

                # Forward pass
                output = model(val_images)
                val_loss = loss_function(output, val_images.view(-1, 1, *image_size))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

    # Save model and logs
    save_model(model, save_path, weights_name, epochs)
    save_log(log_dict, save_path, weights_name, epochs)

    return log_dict



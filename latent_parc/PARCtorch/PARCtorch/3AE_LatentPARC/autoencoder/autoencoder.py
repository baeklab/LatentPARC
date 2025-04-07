import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F


#Base: https://www.digitalocean.com/community/tutorials/convolutional-autoencoder

# Helper Functions #

def init_weights(module):
    """ Initialize weights using Xavier uniform initialization. """
    if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
        
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

# class MLPDecoder(nn.Module):
#     def __init__(self, layers, latent_dim, act_fn=nn.ReLU()):
#         super().__init__()
#         self.output_shape = output_shape 
#         modules = []
#         in_dim = latent_dim
#         for dim in reversed(layers):
#             modules.append(nn.Linear(in_dim, dim))
#             modules.append(act_fn)
#             in_dim = dim
#         # layers.append(nn.Linear(in_dim, output_dim))  # Output layer
#         self.net = nn.Sequential(*modules)

#     def forward(self, x):
#         x = self.net(x)
#         # Reshape to original image dimensions
#         batch_size = x.size(0)
        
#         return x.view(batch_size, *self.output_shape)
    
class MLPDecoder(nn.Module):
    def __init__(self, layers, latent_dim, output_shape=(3, 128, 256), act_fn=nn.ReLU()):
        super().__init__()
        self.output_shape = output_shape  # Now it's properly passed in!
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

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)

### EXTRA CONV LAYER

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
    
    
def train_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
                      device, epochs=10, image_size=(64, 64), n_channels=3, 
                      scheduler=None, noise_fn=None, initial_max_noise=0.16, 
                      n_reduce_factor=0.5, reduce_on=1000, save_path=None, weights_name=None):
    """ Train an autoencoder with optional noise injection. """

    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': []}
    model.to(device)
    model.apply(init_weights)

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
            
            images = F.interpolate(images, size=(image_size[0], image_size[1]), mode='bilinear', align_corners=False) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
                
                val_images = F.interpolate(val_images, size=(image_size[0], image_size[1]), mode='bilinear', align_corners=False) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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






















    
# class Encoder(nn.Module):
#     def __init__(self, layers, latent_dim=128, act_fn=nn.ReLU()):
#         super().__init__()
#         modules = []
#         in_channels = layers[0]
#         for out_channels in layers[1:]:
#             modules.append(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
#             )  # Keep padding=1 for same-sized convolutions
#             modules.append(act_fn)
#             in_channels = out_channels
#         modules.append(
#             nn.Conv2d(layers[-1], latent_dim, kernel_size=3, stride = 2, padding=1)
#         )  # Bottleneck layer
#         self.net = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.net(x)


# # Defining the decoder, no deconv
# class Decoder(nn.Module):
#     def __init__(self, layers, latent_dim=128, act_fn=nn.ReLU()):
#         super().__init__()

#         self.in_channels = layers[-1]
#         self.latent_dim = latent_dim

#         modules = []
#         in_channels = latent_dim #layers[-1]

#         # Initial convolution layer for latent vector
#         # modules.append(nn.Conv2d(latent_dim, in_channels, kernel_size=3, padding=1))

#         # Iteratively create resize-convolution layers
#         for out_channels in reversed(layers): #layers[:-1]
#             modules.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))  # Resizing
#             modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # Convolution
#             modules.append(act_fn)  # Activation function
#             in_channels = out_channels

#         self.conv = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.conv(x)


# # Defining the autoencoder
# class Autoencoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
    
# def add_random_noise(images, min_val=0.0, max_val=0.1):
#     """
#     Add random (uniform) noise to the images.

#     Parameters:
#         images: Tensor of input images.
#         min_val: Minimum value of the noise.
#         max_val: Maximum value of the noise.

#     Returns:
#         Noisy images.
#     """
#     noise = torch.rand_like(images) * (max_val - min_val) + min_val
#     noisy_images = images + noise
#     return torch.clamp(noisy_images, 0.0, 1.0)  # Keep pixel values in [0, 1]

# class LpLoss(torch.nn.Module):
#     def __init__(self, p=10):
#         super(LpLoss, self).__init__()
#         self.p = p

#     def forward(self, input, target):
#         # Compute element-wise absolute difference
#         diff = torch.abs(input - target)
#         # Raise the differences to the power of p, sum them, and raise to the power of 1/p
#         return (torch.sum(diff ** self.p) ** (1 / self.p))

# class ConvolutionalAutoencoder:
#     def __init__(self, autoencoder, optimizer, save_path=None, weights_name=None):
#         self.network = autoencoder
#         self.optimizer = optimizer
#         self.save_path = save_path
#         self.weights_name = weights_name

#     def train(self, loss_function, epochs, image_size, n_channels, device, train_loader, val_loader, scheduler=None, noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5, reduce_on=1000):
#         """
#         Train the network as a denoising autoencoder with PARCv1 acting on the latent space.

#         Parameters:
#             loss_function: The loss function to minimize.
#             epochs: Number of epochs to train.
#             image_size: Tuple representing the image dimensions (height, width).
#             n_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
#             device: The device to run the model on ('cpu' or 'cuda').
#             train_loader: DataLoader for the training dataset.
#             val_loader: DataLoader for the validation dataset.
#             scheduler: Learning rate scheduler
#             noise_fn: A function to add noise to the images. If None, no noise is added.
#             reduce_on: number of epochs on which to reduce noise
#         """
#         # Creating log
#         log_dict = {
#             'training_loss_per_epoch': [],
#             'validation_loss_per_epoch': [],
#         }

#         # Defining weight initialization function
#         def init_weights(module):
#             if isinstance(module, nn.Conv2d):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 module.bias.data.fill_(0.01)
#             elif isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 module.bias.data.fill_(0.01)

#         # Initializing network weights
#         self.network.apply(init_weights)

#         # Setting convnet to training mode
#         self.network.train()
#         self.network.to(device)
        
#         # Initialize max_noise
#         max_noise = initial_max_noise

#         for epoch in range(epochs):
#             print(f'Epoch {epoch + 1}/{epochs}')
#             train_losses = []
#             val_losses = []
            
#             # Update max_noise every 1000 epochs
#             if (epoch + 1) % reduce_on == 0:
#                 max_noise = n_reduce_factor*max_noise
#                 # print(f"Reduced max_noise to {max_noise}")

#             # ------------
#             # TRAINING
#             # ------------
#             print('Training...')
#             self.network.train()
#             for images in tqdm(train_loader):
#                 # Zeroing gradients
#                 self.optimizer.zero_grad()
#                 # Sending images to device
#                 images = images[0][:, 0:n_channels, ...].to(device)
                
#                 # images[:, 2, ...] *= 0.5 #!!!!!!!!!!!!!!!!!!!!TESTING THEORY DIV MS MAG BY 2

#                 # Adding noise to images
#                 noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

#                 # Reconstructing images
#                 output = self.network(noisy_images)
#                 # Computing loss (comparing output with clean images)
#                 loss = loss_function(output, images.view(-1, n_channels, image_size[0], image_size[1]))
#                 # Calculating gradients
#                 loss.backward()
#                 # Optimizing weights
#                 self.optimizer.step()

#                 # Logging loss
#                 train_losses.append(loss.item())

#             # Average training loss for the epoch
#             avg_train_loss = sum(train_losses) / len(train_losses)
#             log_dict['training_loss_per_epoch'].append(avg_train_loss)

#             # ------------
#             # VALIDATION
#             # ------------
#             print('Validating...')
#             self.network.eval()
#             with torch.no_grad():
#                 for val_images in tqdm(val_loader):
#                     # Sending validation images to device
#                     val_images = val_images[0][:, 0:n_channels, ...].to(device)
                    
#                     # val_images[:, 2, ...] *= 0.5 #!!!!!!!!!!!!!!!!!!!!TESTING THEORY DIV MS MAG BY 2


#                     # # Adding noise to validation images
#                     # noisy_val_images = noise_fn(val_images) if noise_fn else val_images

#                     # Reconstructing images
#                     output = self.network(val_images) # noisy_val_images
#                     # Computing validation loss
#                     val_loss = loss_function(output, val_images.view(-1, n_channels, image_size[0], image_size[1]))
#                     # Logging loss
#                     val_losses.append(val_loss.item())

#             # Average validation loss for the epoch
#             avg_val_loss = sum(val_losses) / len(val_losses)
#             log_dict['validation_loss_per_epoch'].append(avg_val_loss)

#             print(f"Epoch {epoch + 1}: Training Loss: {round(avg_train_loss, 4)} Validation Loss: {round(avg_val_loss, 4)}")
            
#             if scheduler:
#                 scheduler.step(val_loss)
#                 # print(f"Learning rate adjusted to: {scheduler.get_last_lr()}")


#         # ------------
#         # SAVE WEIGHTS
#         # ------------
#         if self.save_path:
#             save_file = f"{self.save_path}/{self.weights_name}_{epochs}.pth"
#             torch.save(self.network.state_dict(), save_file)
#             print(f"Model weights saved to {save_file}")

#         # Save log_dict to JSON file
#         if self.save_path:
#             log_file = f"{self.save_path}/{self.weights_name}_{epochs}.json"
#             with open(log_file, 'w') as f:
#                 json.dump(log_dict, f)

#         return log_dict

#     def autoencode(self, x):
#         return self.network(x)

#     def encode(self, x):
#         encoder = self.network.encoder
#         return encoder(x)

#     def decode(self, x):
#         decoder = self.network.decoder
#         return decoder(x)





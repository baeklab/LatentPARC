import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from torchvision.utils import make_grid
import numpy as np
# import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from PARCv1.differentiator import *
from autoencoder.autoencoder import *

# miscellaneous
class LpLoss(torch.nn.Module):
    def __init__(self, p=10):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, input, target):
        # Compute element-wise absolute difference
        diff = torch.abs(input - target)
        # Raise the differences to the power of p, sum them, and raise to the power of 1/p
        return (torch.sum(diff ** self.p) ** (1 / self.p))

# Defining LatentPARC
class lp_model(nn.Module):
    def __init__(self, encoder, decoder, differentiator, integrator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.differentiator = differentiator
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        '''
        x: initial condition input data (T, P, ms) (for rollout pred, can be z input)
        n_ts: number of ts to propogate in latent space
        
        (when n_ts = 0 just AE no differentiator)
        can load in target or ic or target to x depending on what
        right now only n_ts = 0 or 1 enabled? with more you get rollout training/pred
        to enable more rollout during training, will need to change data loader to support rollout? 
        enable more than 
        '''
        # encoded = self.encoder(x)
        # next_t, _ = self.integrator(self.differentiator, 0.0, encoded, 0.1)
        # decoded = self.decoder(next_t)
        
        if mode == 'train':
            if n_ts==0:
                z = self.encoder(x)
                decoded = self.decoder(z)

            elif n_ts==1:
                z_init = self.encoder(x)
                z, _ = self.integrator(self.differentiator, 0.0, z_init, 0.1)
                decoded = self.decoder(z)
                
        elif mode == 'pred':
            z_list = []
            decoded_list = []

            z_i = self.encoder(x)
            z_list.append(z_i)

            d = self.decoder(z_i)
            decoded_list.append(d)

            for i in range(n_ts - 1):
                z_i, _ = self.integrator(self.differentiator, 0.0, z_i, 0.1)
                z_list.append(z_i)

                d = self.decoder(z_i)
                decoded_list.append(d)

            z = torch.stack(z_list, dim=0)  # Stack along new dimension
            decoded = torch.stack(decoded_list, dim=0)  # Ensure same shape
        
        return z, decoded


class LatentPARC:
    def __init__(self, lp_model, optimizer, save_path=None, weights_name=None):
        self.network = lp_model
        self.optimizer = optimizer
        self.save_path = save_path
        self.weights_name = weights_name

    def train(self, loss_function, epochs, image_size, n_channels, device, train_loader, val_loader, scheduler=None, noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5, reduce_on=1000):
        """
        Train the network as a denoising autoencoder.

        Parameters:
            loss_function: The loss function to minimize.
            epochs: Number of epochs to train.
            image_size: Tuple representing the image dimensions (height, width).
            n_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            device: The device to run the model on ('cpu' or 'cuda').
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            scheduler: Learning rate scheduler
            noise_fn: A function to add noise to the images. If None, no noise is added.
            reduce_on: number of epochs on which to reduce noise
        """
        # Creating log
        log_dict = {
            'training_loss_per_epoch': [],
            'validation_loss_per_epoch': [],
        }

        # Defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

        # Initializing network weights
        self.network.apply(init_weights)

        # Setting convnet to training mode
        self.network.train()
        self.network.to(device)
        
        # Initialize max_noise
        max_noise = initial_max_noise

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_losses = []
            val_losses = []
            
            # Update max_noise every 1000 epochs
            if (epoch + 1) % reduce_on == 0:
                max_noise = n_reduce_factor*max_noise
                # print(f"Reduced max_noise to {max_noise}")

            # ------------
            # TRAINING
            # ------------
            print('Training...')
            self.network.train()
            for images in tqdm(train_loader):
                # Zeroing gradients
                self.optimizer.zero_grad()
                ic, _, _, target = images # ic is initial condition state, target is the next timestep evolution of ic !triple check this
                
                ic = ic[:, 0:n_channels, ...].to(device)
                target = target.squeeze(0)[:, 0:n_channels, ...].to(device)
                
                # HALVE MICROSTRUCTURE
                ic[:, 2, ...] *= 0.5 #!!!!!!!!!!!!!!!!!!!!TESTING THEORY DIV MS MAG BY 2
                target[:, 2, ...] *= 0.5 #!!!!!!!!!!!!!!!!!!!!TESTING THEORY DIV MS MAG BY 2

                # Adding noise to images
                noisy_ic = noise_fn(ic, max_val=max_noise) if noise_fn else ic
                noisy_target = noise_fn(target, max_val=max_noise) if noise_fn else target

                # Reconstructing images
                # output = self.network(noisy_ic)
                _, x_bar = self.network(noisy_ic, n_ts=0) #maybe for efficiency, save z here and feed into next line if just differentiator?? only need to encode once even if doing rollout
                z_hat, x_hat = self.network(noisy_ic, n_ts=1)
                z_hat_next, _ = self.network(noisy_target, n_ts=0)
                
                # Computing loss (comparing output with clean images)
                # loss = loss_function(output, target.view(-1, n_channels, image_size[0], image_size[1])) 
                
                L_r = loss_function(x_bar, ic.view(-1, n_channels, image_size[0], image_size[1]))
                L_d1 = loss_function(z_hat, z_hat_next)
                L_d2 = loss_function(x_hat, target.view(-1, n_channels, image_size[0], image_size[1]))
                
                loss = L_r + L_d1 + L_d2
                
                # Calculating gradients
                loss.backward()
                # Optimizing weights
                self.optimizer.step()

                # Logging loss
                train_losses.append(loss.item())

            # Average training loss for the epoch
            avg_train_loss = sum(train_losses) / len(train_losses)
            log_dict['training_loss_per_epoch'].append(avg_train_loss)

            # ------------
            # VALIDATION
            # ------------
            print('Validating...')
            self.network.eval()
            with torch.no_grad():
                for val_images in tqdm(val_loader):
                    
                    # Sending validation images to device
                    val_ic, _, _, val_target = val_images
                    val_ic = val_ic[:, 0:n_channels, ...].to(device)
                    val_target = val_target.squeeze(0)[:, 0:n_channels, ...].to(device)
                    
                    # HALVE MICROSTRUCTURE
                    val_ic[:, 2, ...] *= 0.5 #!!!!!!!!!!!!!!!!!!!!TESTING THEORY DIV MS MAG BY 2
                    val_target[:, 2, ...] *= 0.5 #!!!!!!!!!!!!!!!!!!!!TESTING THEORY DIV MS MAG BY 2

                    # Reconstructing images  
                    _, x_bar = self.network(val_ic, n_ts=0) 
                    z_hat, x_hat = self.network(val_ic, n_ts=1)
                    z_hat_next, _ = self.network(val_target, n_ts=0)

                    # Computing validation loss
                    L_r = loss_function(x_bar, val_ic.view(-1, n_channels, image_size[0], image_size[1]))
                    L_d1 = loss_function(z_hat, z_hat_next)
                    L_d2 = loss_function(x_hat, val_target.view(-1, n_channels, image_size[0], image_size[1]))

                    val_loss = L_r + L_d1 + L_d2
                    
                    # Logging loss
                    val_losses.append(val_loss.item())

            # Average validation loss for the epoch
            avg_val_loss = sum(val_losses) / len(val_losses)
            log_dict['validation_loss_per_epoch'].append(avg_val_loss)

            print(f"Epoch {epoch + 1}: Training Loss: {round(avg_train_loss, 4)} Validation Loss: {round(avg_val_loss, 4)}")
            
            if scheduler:
                scheduler.step(val_loss)
                # print(f"Learning rate adjusted to: {scheduler.get_last_lr()}")


        # ------------
        # SAVE WEIGHTS
        # ------------
        if self.save_path:
            save_file = f"{self.save_path}/{self.weights_name}_{epochs}.pth"
            torch.save(self.network.state_dict(), save_file)
            print(f"Model weights saved to {save_file}")

        # Save log_dict to JSON file
        if self.save_path:
            log_file = f"{self.save_path}/{self.weights_name}_{epochs}.json"
            with open(log_file, 'w') as f:
                json.dump(log_dict, f)

        return log_dict

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)

    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)
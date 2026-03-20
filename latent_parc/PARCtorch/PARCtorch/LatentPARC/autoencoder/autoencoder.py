import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
import os

# # Helper Functions #
# def save_model(model, save_path, weights_name, epochs):
#     """ Save model weights to file. """
#     if save_path:
#         save_file = f"{save_path}/{weights_name}_{epochs}.pth"
#         torch.save(model.state_dict(), save_file)
#         print(f"Model weights saved to {save_file}")
        
# def save_log(log_dict, save_path, weights_name, epochs):
#     """ Save training logs as JSON. """
#     if save_path:
#         log_file = f"{save_path}/{weights_name}_{epochs}.json"
#         with open(log_file, 'w') as f:
#             json.dump(log_dict, f)

def save_model_and_logs(
    model,
    optimizer,
    save_path,
    weights_name,
    log_dict,
    epoch,
    scheduler=None,
    current_max_noise=None,
    config=None,
    noise_config=None,
    extra_state=None,
):
    """
    Save full training checkpoint + logs.
    """

    if not save_path:
        return

    os.makedirs(save_path, exist_ok=True)

    # --- Core checkpoint ---
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # --- Scheduler ---
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # --- Current training state (dynamic things) ---
    if current_max_noise is not None:
        checkpoint["current_max_noise"] = current_max_noise

    # --- Configs (static + semi-static) ---
    if config is not None:
        checkpoint["config"] = config

    if noise_config is not None:
        # Optionally inject current noise for readability
        noise_config = dict(noise_config)  # avoid mutating original
        if current_max_noise is not None:
            noise_config["current_max_noise"] = current_max_noise

        checkpoint["noise_config"] = noise_config

    # --- Extra ---
    if extra_state is not None:
        checkpoint["extra_state"] = extra_state

    # --- Save checkpoint ---
    ckpt_path = os.path.join(save_path, f"{weights_name}_{epoch + 1}.pth")
    torch.save(checkpoint, ckpt_path)

    # --- Save logs ---
    log_path = os.path.join(save_path, f"{weights_name}_{epoch + 1}.json")
    with open(log_path, "w") as f:
        json.dump(log_dict, f, indent=2)

    print(f"Checkpoint + logs saved to {save_path}")
            
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
    
# Define Modules #
    
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
    
    
def train_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
                      device, epochs=3000, save_every_epochs = 100, image_size=(64, 64), n_channels=3, 
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

                # Forward pass
                output = model(val_images)
                val_loss = loss_function(output, val_images.view(-1, n_channels, *image_size))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

        if (epoch + 1) % save_every_epochs == 0:
            save_model_and_logs(
                model=model,
                optimizer=optimizer,
                save_path=save_path,
                weights_name=weights_name,
                log_dict=log_dict,
                epoch=epoch,
                scheduler=scheduler,
                current_max_noise=max_noise,
                
                config={ # just for settings info, not needed for commencing training
                    "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
                    "current_lr": optimizer.param_groups[0]["lr"],
                    "image_size": image_size,
                    "n_channels": n_channels,
                },
            
                noise_config={ # just for settings info, not needed for commencing training
                    "initial_max_noise": initial_max_noise,
                    "n_reduce_factor": n_reduce_factor,
                    "reduce_on": reduce_on,
                },
            )

    # Save model and logs
    
    save_model_and_logs(
                model=model,
                optimizer=optimizer,
                save_path=save_path,
                weights_name=weights_name,
                log_dict=log_dict,
                epoch=epoch,
                scheduler=scheduler,
                current_max_noise=max_noise,
            
                config={
                    "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
                    "current_lr": optimizer.param_groups[0]["lr"],
                    "image_size": image_size,
                    "n_channels": n_channels,
                },
            
                noise_config={
                    "initial_max_noise": initial_max_noise,
                    "n_reduce_factor": n_reduce_factor,
                    "reduce_on": reduce_on,
                },
            )
    
    return log_dict


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



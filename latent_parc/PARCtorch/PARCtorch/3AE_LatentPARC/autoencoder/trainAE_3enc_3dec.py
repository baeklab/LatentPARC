import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torchvision.utils import make_grid
from autoencoder import *
from torch.optim import Adam
import json

# LOAD IN DATA
# Add the root directory (PARCTorch) to the system path
path = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Your initial path
new_path = os.path.dirname(path)  # Remove the last folder

sys.path.append(new_path)
from data.normalization import compute_min_max

data_dirs = [
    "/project/vil_baek/data/physics/PARCTorch/HMX/train",
    "/project/vil_baek/data/physics/PARCTorch/HMX/test",
]
output_file = new_path + "/data/hmx_min_max.json"
compute_min_max(data_dirs, output_file)

# CREATE DATA LOADERS
from torch.utils.data import DataLoader, random_split
import logging
from data.dataset import (
    GenericPhysicsDataset,
    custom_collate_fn,
)
from utilities.viz import visualize_channels, save_gifs_with_ground_truth

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Example configuration for HMX dataset
data_dir_train = "/project/vil_baek/data/physics/PARCTorch/HMX/train"  # Replace with your actual train directory path
data_dir_test = "/project/vil_baek/data/physics/PARCTorch/HMX/test"  # Replace with your actual test directory path
n_ts = 2
# Path to the min_max.json file
min_max_path = os.path.join(new_path, "data", "hmx_min_max.json")  # Correct path
batch_size = 32
validation_split = 0.05  # 5% for validation

# Initialize the dataset
train_dataset = GenericPhysicsDataset(
    data_dirs=[data_dir_train],
    future_steps=n_ts-1,
    min_max_path=min_max_path,
    vflip_prob=0.5, # !!!!!!!!!!!!!!!!!!!! remember to change
)

# Calculate the size of the validation set
validation_size = int(len(train_dataset) * validation_split)
train_size = len(train_dataset) - validation_size

# Perform the split
train_subset, val_subset = random_split(train_dataset, [train_size, validation_size])

# Create DataLoader for training and validation datasets
train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle the training data
    num_workers=1,
    pin_memory=True,
    collate_fn=custom_collate_fn,
)

val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle validation data
    num_workers=1,
    pin_memory=True,
    collate_fn=custom_collate_fn,
)

# DEFINE AUTOENCODER + TRAINING
# where to save weights
save_path="/sfs/gpfs/tardis/home/pdy2bw/Research/LatentPARC/latent_parc/PARCtorch/PARCtorch/3AE_LatentPARC/autoencoder"
weights_name="popped_dec_vertflip_layers_1_2_latent_4_MAE_DE_Nmax16_nrf8_redon10_LRplateau_e3_factor8_pat10"

# model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define layer sizes and initialize encoder/decoder
layer_sizes = [1, 2]
latent_dim = 4

encoder_T = Encoder(layers=layer_sizes, latent_dim=latent_dim)
encoder_P = Encoder(layers=layer_sizes, latent_dim=latent_dim)
encoder_M = Encoder(layers=layer_sizes, latent_dim=latent_dim)

decoder_T = Decoder(layers=layer_sizes, latent_dim=latent_dim)
decoder_P = Decoder(layers=layer_sizes, latent_dim=latent_dim)
decoder_M = Decoder(layers=layer_sizes, latent_dim=latent_dim)

AE_T = Autoencoder(encoder_T, decoder_T)
AE_P = Autoencoder(encoder_P, decoder_P)
AE_M = Autoencoder(encoder_M, decoder_M)

criterion = torch.nn.L1Loss().cuda()

optimizer_T = Adam(AE_T.parameters(), lr=1e-3)
optimizer_P = Adam(AE_P.parameters(), lr=1e-3)
optimizer_M = Adam(AE_M.parameters(), lr=1e-3)

scheduler_T = ReduceLROnPlateau(optimizer_T, mode='min', factor=0.8, patience=10, verbose=True)
scheduler_P = ReduceLROnPlateau(optimizer_P, mode='min', factor=0.8, patience=10, verbose=True)
scheduler_M = ReduceLROnPlateau(optimizer_M, mode='min', factor=0.8, patience=10, verbose=True)

model_T = ConvolutionalAutoencoder(AE_T, optimizer_T, device, save_path, f"{weights_name}_T")
model_P = ConvolutionalAutoencoder(AE_P, optimizer_P, device, save_path, f"{weights_name}_P")
model_M = ConvolutionalAutoencoder(AE_M, optimizer_M, device, save_path, f"{weights_name}_M")

epochs = 1500

log_T = train_individual_autoencoder(model_T.network, optimizer_T, criterion, train_loader, val_loader, 
                             device=device, epochs=epochs, image_size=[128, 256], channel_index=0, 
                             scheduler=scheduler_T, noise_fn=add_random_noise, initial_max_noise=0.16,
                             n_reduce_factor=0.8, reduce_on=10, save_path=save_path,
                             weights_name=f"{weights_name}_T")

log_P = train_individual_autoencoder(model_P.network, optimizer_P, criterion, train_loader, val_loader, 
                             device=device, epochs=epochs, image_size=[128, 256], channel_index=1, 
                             scheduler=scheduler_P, noise_fn=add_random_noise, initial_max_noise=0.16,
                             n_reduce_factor=0.8, reduce_on=10, save_path=save_path,
                             weights_name=f"{weights_name}_P")

log_M = train_individual_autoencoder(model_M.network, optimizer_M, criterion, train_loader, val_loader, 
                             device=device, epochs=epochs, image_size=[128, 256], channel_index=2, 
                             scheduler=scheduler_M, noise_fn=add_random_noise, initial_max_noise=0.16, 
                             n_reduce_factor=0.8, reduce_on=10, save_path=save_path,
                             weights_name=f"{weights_name}_M")

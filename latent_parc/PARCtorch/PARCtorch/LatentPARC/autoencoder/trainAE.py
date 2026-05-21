import torch
import torch.nn as nn
import os
import sys
import logging
from torch.optim.lr_scheduler import StepLR

from autoencoder_backbones import *
from torch.optim import Adam
from utils import add_random_noise
from train_scripts import train_autoencoder

# Add the root directory (PARCTorch) to the system path
base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(base_path)

# LOAD IN DATA
from data.normalization import compute_min_max
data_dirs = [
    # "/project/vil_baek/data/physics/PARCTorch/HMX/train",
    # "/project/vil_baek/data/physics/PARCTorch/HMX/test",
    "/standard/sds_baek_energetic/data/physics/Shahab_Latent_PARC/Processed_data/PARCtorch_meso_scale/train",
    "/standard/sds_baek_energetic/data/physics/Shahab_Latent_PARC/Processed_data/PARCtorch_meso_scale/test",
]
output_file = os.path.join(base_path, "data", "hmx_min_max.json")
compute_min_max(data_dirs, output_file)

# CREATE DATA LOADERS
from torch.utils.data import DataLoader, random_split
from data.dataset import (
    GenericPhysicsDataset,
    custom_collate_fn,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Example configuration for HMX dataset
# data_dir_train = "/project/vil_baek/data/physics/PARCTorch/HMX/train"  # Replace with your actual train directory path
data_dir_train = "/standard/sds_baek_energetic/data/physics/Shahab_Latent_PARC/Processed_data/PARCtorch_meso_scale/train"  # Replace with your actual train directory path

n_ts = 2
# Path to the min_max.json file
min_max_path = os.path.join(base_path, "data", "hmx_min_max.json")  # Correct path
batch_size = 1
validation_split = 0.05  # 20% for validation

# Initialize the dataset
train_dataset = GenericPhysicsDataset(
    data_dirs=[data_dir_train],
    future_steps=n_ts-1,
    min_max_path=min_max_path,
    # vflip_prob=0.5,
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

# ------------------------
# Training Hyperparameters
# ------------------------
save_path = os.path.join(base_path, "LatentPARC", "autoencoder")
weights_name = "clarenceAE_plusStepLR"
weights_path = os.path.join(save_path, f"{weights_name}.pth")  # optional, check for existing weights to load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs=3000
save_every_epochs=500
# image_size=[128, 256]
image_size=[160, 240]
n_channels=3

initial_max_noise=0.16
n_reduce_factor=0.8
reduce_noise_on=500

layer_sizes=[3, 8]
latent_dim=8

# ------------------------
# Model setup
# ------------------------
encoder = Encoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
decoder = Decoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
autoencoder = Autoencoder(encoder, decoder).to(device)

criterion = torch.nn.L1Loss().to(device)
optimizer = Adam(autoencoder.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

# ------------------------
# Optionally load from checkpoint
# ------------------------
if os.path.exists(weights_path):
    print(f"Loading checkpoint from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)

    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # restore noise state if available
    initial_max_noise = checkpoint.get("current_max_noise", initial_max_noise)
else:
    print("No checkpoint found, starting from scratch.")

# ------------------------
# Wrap in ConvolutionalAutoencoder
# ------------------------
model = ConvolutionalAutoencoder(autoencoder, optimizer, device, save_path, weights_name)

# ------------------------
# Train
# ------------------------
log_dict = train_autoencoder(
    model.network,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device=device,
    epochs=epochs,       
    save_every_epochs=save_every_epochs,
    image_size=image_size,
    n_channels=n_channels,
    scheduler=scheduler,
    noise_fn=add_random_noise,
    initial_max_noise=initial_max_noise,
    n_reduce_factor=n_reduce_factor,
    reduce_on=reduce_noise_on,
    save_path=save_path,
    weights_name=weights_name
)

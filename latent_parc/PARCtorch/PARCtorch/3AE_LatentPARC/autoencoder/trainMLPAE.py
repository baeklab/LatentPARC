import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

def set_seed(seed: int):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)
    
    # Set the seed for PyTorch (GPU), if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    
    # Ensure deterministic behavior in certain cases
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This can be set to True for non-deterministic algorithms (faster on some hardware)

set_seed(42)

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

import os
import torch
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
future_steps = 3
# Path to the min_max.json file
min_max_path = os.path.join(new_path, "data", "hmx_min_max.json")  # Correct path
batch_size = 32
validation_split = 0.05  # 20% for validation

# Initialize the dataset
train_dataset = GenericPhysicsDataset(
    data_dirs=[data_dir_train],
    future_steps=future_steps,
    min_max_path=min_max_path,
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

# Optionally, create DataLoader for test dataset
test_dataset = GenericPhysicsDataset(
    data_dirs=[data_dir_test],
    future_steps=future_steps,
    min_max_path=min_max_path,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle test data
    num_workers=1,
    pin_memory=True,
    collate_fn=custom_collate_fn,
)

# Loss funcs

# class LpLoss(torch.nn.Module):
#     def __init__(self, p=10):
#         super(LpLoss, self).__init__()
#         self.p = p

#     def forward(self, input, target):
#         # Compute element-wise absolute difference
#         diff = torch.abs(input - target)
#         # Summing over all dimensions except the batch dimension
#         norm = torch.sum(diff**self.p, dim=tuple(range(1, diff.ndim)))**(1/self.p)
#         return torch.mean(norm)  # Compute batch mean

# DEFINE AUTOENCODER + TRAINING
# where to save weights
save_path="/sfs/gpfs/tardis/home/pdy2bw/Research/LatentPARC/latent_parc/PARCtorch/PARCtorch/3AE_LatentPARC/autoencoder"
weights_name="MLP_AE_64by128_L1_loss_layers_3_12288_6144_latent_3072_DE_Nmax16_nrf8_redon200_LRstep_e3_factor8_stepsize200"

# model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

height = 64
width = 128

output_shape = (3, height, width)

# layer_sizes = [3 * (height * width), 8 * (height//2) * (width//2)]
# latent_dim = 8 * (height//4) * (width//4)

layer_sizes = [3 * (height * width), 12288, 6144]
latent_dim = 3072

encoder = MLPEncoder(layers=layer_sizes, latent_dim=latent_dim).to(device)
decoder = MLPDecoder(layers=layer_sizes, latent_dim=latent_dim, output_shape=output_shape).to(device)

# Initialize autoencoder
autoencoder = Autoencoder(encoder, decoder).to(device)

# Loss Func
criterion = torch.nn.L1Loss().cuda()
# criterion = torch.nn.MSELoss().cuda()
# criterion = LpLoss(p=7).cuda()

# criterion = nn.MSELoss()
optimizer = Adam(autoencoder.parameters(), lr=1e-3)

# Define learning rate scheduler
scheduler = StepLR(optimizer, step_size=200, gamma=0.8)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)


#  training model - one save
model = ConvolutionalAutoencoder(autoencoder, optimizer, device, save_path, weights_name)

log_dict = train_autoencoder(model.network, optimizer, criterion, train_loader, val_loader, 
                             device=device, epochs=1000, image_size=[64, 128], n_channels=3, 
                             scheduler=scheduler, noise_fn=add_random_noise, initial_max_noise=0.16, 
                             n_reduce_factor=0.8, reduce_on=200, 
                             save_path=save_path, weights_name=weights_name)

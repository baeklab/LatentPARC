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
from torch.optim import Adam
import json
from torchsummary import summary

from autoencoder.autoencoder import *
from PARCv1.differentiator import *
from PARCtorch.integrator.rk4 import *
from PARCtorch.integrator.numintegrator import *
from LatentPARC_model import *

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


# get PARCtorch path

path = os.path.abspath(os.path.join(os.getcwd(), "..")) 
print(path)

# Add the root directory (PARCTorch) to the system path
sys.path.append(path)
from data.normalization import compute_min_max

# Get single void HMX data

data_dirs = [
    "/project/vil_baek/data/physics/PARCTorch/HMX/train",
    "/project/vil_baek/data/physics/PARCTorch/HMX/test",
]
output_file = path + "/data/hmx_min_max.json"
compute_min_max(data_dirs, output_file)

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
future_steps = 1
# Path to the min_max.json file
min_max_path = os.path.join(path, "data", "hmx_min_max.json")  # Correct path
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




#  TRAINING PARAMS + RUN

# where to save weights
save_path="/sfs/gpfs/tardis/home/pdy2bw/Research/LatentPARC/latent_parc/PARCtorch/PARCtorch/3AE_LatentPARC"
weights_name="layers_2_4_latent_4_1_2_latent_2_decoder_3_6_LP_2plus1_1decoder_Nmax16_nrf8_redon500_LRstep_e3_factor8_stepsize200_nts1"

# model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define layer sizes and initialize encoder/decoder
layer_sizes_TP = [2, 4]
latent_dim_TP = 4
layer_sizes_M = [1, 2]
latent_dim_M = 2

layer_sizes_decoder = [3, 6]
latent_dim = latent_dim_TP + latent_dim_M

encoder_TP = Encoder(layers=layer_sizes_TP, latent_dim=latent_dim_TP).to(device)
encoder_M = Encoder(layers=layer_sizes_M, latent_dim=latent_dim_M).to(device)
decoder = Decoder(layers=layer_sizes_decoder, latent_dim=latent_dim).to(device)
differentiator = Differentiator(latent_dim=latent_dim)
integrator = RK4().to(device)  # step size may be hyper-param of interest

# Initialize LatentPARC
model_init = lp_model_2plus1_1decoder(encoder_TP, encoder_M, decoder, differentiator, integrator).to(device)

#Loss Function
criterion = torch.nn.L1Loss().to(device)
# criterion = LpLoss(p=10).cuda()

# #### BEGIN PERCEPTUAL LOSS BLOCK
# # Initialize Perceptual Loss
# feature_extractor = FeatureExtractor(layers=[1, 3, 6]).to(device)
# perceptual_loss = PerceptualLoss(feature_extractor).to(device)

# # Perceptual Loss + L1 Combined Criterion
# def combined_loss(output, target):
#     l1_loss = criterion(output, target)  # Existing L1 loss
#     # Check if the number of channels is 3 before computing perceptual loss
#     if target.shape[1] == 3:  
#         p_loss = perceptual_loss(output, target)  # Perceptual loss
#     else:
#         p_loss = 0  # No perceptual loss for latent space data (more than 3 channels)
#     total_loss = l1_loss + 0.1 * p_loss  # Weight perceptual loss (adjust 0.1 as needed)
#     return total_loss
# #### END PERCEPTUAL LOSS BLOCK

# criterion = nn.MSELoss()
optimizer = Adam(model_init.parameters(), lr=1e-3)

# Define learning rate scheduler
# scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

#  training model
model = LatentPARC(model_init, optimizer, save_path, weights_name)

log_dict = model.train(loss_function=criterion, epochs=3000, image_size = [128, 256], 
                       n_channels=3, device=device, train_loader=train_loader, val_loader=val_loader,
                       scheduler=scheduler, noise_fn=add_random_noise, initial_max_noise=0.16, 
                       n_reduce_factor=0.8, ms_reduce_factor=0, reduce_on=200, loss_weights=[1.0,1.0,1.0])
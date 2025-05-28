# Standard libraries
import os
import sys
import json
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

# Third-party libraries
from tqdm import tqdm
from torchsummary import summary

# PyTorch core
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# PyTorch learning rate schedulers
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# PyTorch utilities
from torchvision.utils import make_grid

# Project-specific modules
from autoencoder.autoencoder import *
from PARCv1.differentiator import *
from PARCtorch.integrator.rk4 import *
from PARCtorch.integrator.numintegrator import *
from LatentPARC_model import *


# get PARCtorch path
path = os.path.abspath(os.path.join(os.getcwd(), "..")) 
print(path)

# Add the root directory (PARCTorch) to the system path
sys.path.append(path)
from data.dataset import (
    GenericPhysicsDataset,
    custom_collate_fn,
)
from utilities.viz import visualize_channels, save_gifs_with_ground_truth
from data.normalization import compute_min_max

# Get single void HMX data
data_dirs = [
    "/project/vil_baek/data/physics/PARCTorch/HMX/train",
    "/project/vil_baek/data/physics/PARCTorch/HMX/test",
]
output_file = path + "/data/hmx_min_max.json"
compute_min_max(data_dirs, output_file)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Example configuration for HMX dataset
data_dir_train = "/project/vil_baek/data/physics/PARCTorch/HMX/train"  # Replace with your actual train directory path
data_dir_test = "/project/vil_baek/data/physics/PARCTorch/HMX/test"  # Replace with your actual test directory path

# Path to the min_max.json file
min_max_path = os.path.join(path, "data", "hmx_min_max.json")  # Correct path


n_ts = 5 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DONT FORGET TO CHANGE


batch_size = 32
validation_split = 0.05  # 5% for validation

# Initialize the dataset
train_dataset = GenericPhysicsDataset(
    data_dirs=[data_dir_train],
    future_steps=n_ts-1,
    min_max_path=min_max_path,
    vflip_prob=0.5,  #!!!!! vert flip enabled
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

#  TRAINING PARAMS + RUN

# where to save weights
save_path="/sfs/gpfs/tardis/home/pdy2bw/Research/LatentPARC/latent_parc/PARCtorch/PARCtorch/3AE_LatentPARC"
weights_name="rollout_train_nts_5_FROZEN_3enc_3dec_vertflip_ReLU_normal_conv_ReLU_out_layers_1_2_latent_6_NONOISE_LRplateau_factor8_pat10"

# model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define layer sizes and initialize encoder/decoder
layer_sizes_T = [1, 2]
latent_dim_T = 2
layer_sizes_P = [1, 2]
latent_dim_P = 2
layer_sizes_M = [1, 2]
latent_dim_M = 2

latent_dim = latent_dim_T + latent_dim_P + latent_dim_M # 12

encoder_T = Encoder(layers=layer_sizes_T, latent_dim=latent_dim_T).to(device)
encoder_P = Encoder(layers=layer_sizes_P, latent_dim=latent_dim_P).to(device)
encoder_M = Encoder(layers=layer_sizes_M, latent_dim=latent_dim_M).to(device)
decoder_T = Decoder(layers=layer_sizes_T, latent_dim=latent_dim_T).to(device)
decoder_P = Decoder(layers=layer_sizes_P, latent_dim=latent_dim_P).to(device)
decoder_M = Decoder(layers=layer_sizes_M, latent_dim=latent_dim_M).to(device)

differentiator = Differentiator(latent_dim=latent_dim)
integrator = RK4().to(device)  # step size may be hyper-param of interest

# ONLY RUN IF LOADING PRETRAINED WEIGHTS TO AE
# !!!! If this works, modify AE code to save encoder and decoder weights separate for simplicity
AE_weights_path = "autoencoder/3enc_3dec_vertflip_ReLU_normal_conv_ReLU_out_layers_1_2_latent_6_MAE_DE_Nmax16_nrf8_redon500_LRstep_e3_factor8_stepsize200_3000.pth"
ckpt = torch.load(AE_weights_path, map_location=device)

# Load encoder weights
encoder_T.load_state_dict({k.replace('encoderT.', ''): v for k, v in ckpt.items() if k.startswith('encoderT.')})
encoder_P.load_state_dict({k.replace('encoderP.', ''): v for k, v in ckpt.items() if k.startswith('encoderP.')})
encoder_M.load_state_dict({k.replace('encoderM.', ''): v for k, v in ckpt.items() if k.startswith('encoderM.')})

# Load decoder weights
decoder_T.load_state_dict({k.replace('decoderT.', ''): v for k, v in ckpt.items() if k.startswith('decoderT.')})
decoder_P.load_state_dict({k.replace('decoderP.', ''): v for k, v in ckpt.items() if k.startswith('decoderP.')})
decoder_M.load_state_dict({k.replace('decoderM.', ''): v for k, v in ckpt.items() if k.startswith('decoderM.')})

################################################
# OPTIONAL: Freeze AE weights
for module in [encoder_T, encoder_P, encoder_M, decoder_T, decoder_P, decoder_M]:
    for param in module.parameters():
        param.requires_grad = False
################################################

# Initialize LatentPARC
model_init = lp_model_3encoder_3decoder(encoder_T, encoder_P, encoder_M, decoder_T,
                                        decoder_P, decoder_M, differentiator, 
                                        integrator).to(device)

#Loss Function
criterion = torch.nn.L1Loss().to(device)
# criterion = LpLoss(p=10).cuda()
# criterion = nn.MSELoss()

optimizer = Adam(model_init.parameters(), lr=1e-3)

# Define learning rate scheduler
# scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)

#  training model
model = LatentPARC(model_init, optimizer, save_path, weights_name)

log_dict = model.train(criterion, epochs=1000, image_size = [128, 256], n_channels=3, device=device, 
                       train_loader=train_loader, val_loader=val_loader, scheduler=scheduler,
                       noise_fn=add_random_noise, initial_max_noise=0, n_reduce_factor=0.8, 
                       ms_reduce_factor=0, reduce_on=200, loss_weights=[1.0,1.0,1.0], 
                       mode="rollout_train")
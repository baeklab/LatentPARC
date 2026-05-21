# =============================================================================
# LatentPARC Training Script
# =============================================================================

# ── Standard Libraries ────────────────────────────────────────────────────────
import os
import sys
import json
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

# ── Third-Party Libraries ─────────────────────────────────────────────────────
from tqdm import tqdm

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Addressing numerical instability issue that was introduced in move from torch 2.4 to 2.7
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ── Project-Specific Modules ──────────────────────────────────────────────────
BASE_PATH = "/sfs/gpfs/tardis/home/pdy2bw/Research/LatentPARC_new" # path to your LatentPARC directory folder

sys.path.append(f"{BASE_PATH}/LatentPARC/latent_parc/PARCtorch")
sys.path.append(f"{BASE_PATH}/LatentPARC/latent_parc/PARCtorch/PARCtorch/LatentPARC")

path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(path)

from autoencoder.autoencoder_backbones import *
from autoencoder.utils import add_random_noise
from PARCv1.differentiator import *
from PARCtorch.integrator.rk4 import *
from PARCtorch.integrator.numintegrator import *
from model import *
from utils import *
from data.normalization import compute_min_max
from data.dataset import GenericPhysicsDataset, custom_collate_fn
from utilities.viz import visualize_channels, save_gifs_with_ground_truth

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# ── Data ──────────────────────────────────────────────────────────────────────
N_TIMESTEPS         = 5          # 2 = single future timestep (target); increase for multi-step rollout
BATCH_SIZE          = 1
VALIDATION_SPLIT    = 0.05       # Fraction of training data used for validation
IMAGE_SIZE          = [128, 256] # [680, 1000]
N_CHANNELS          = 3

# ── Model Architecture ────────────────────────────────────────────────────────
LAYER_SIZES         = [3, 8]     # Channel sizes for encoder/decoder layers
LATENT_DIM          = 8          # Dimensionality of the latent space

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS              = 200
SAVE_RATE           = 50        # Save state dict every X epochs
LEARNING_RATE       = 1e-4
LOSS_WEIGHTS        = [1.0, 1.0, 1.0]    # [reconstruction, latent, rollout]
MODE                = "rollout_train"

# ── Noise Schedule ────────────────────────────────────────────────────────────
INITIAL_MAX_NOISE   = 0          # Starting noise magnitude (0 = no noise)
N_REDUCE_FACTOR     = 0          # Factor by which noise is reduced
MS_REDUCE_FACTOR    = 0          # Multi-step noise reduce factor
REDUCE_ON           = 50         # Epoch interval for noise reduction

# ── Scheduler and Loss ─────────────────────────────────────────────
scheduler = "plateau" # options None, step, plateau

loss_function = torch.nn.L1Loss().to(device)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIRS = [
    "/project/vil_baek/data/physics/PARCTorch/HMX/train",  # SINGLE PORE
    # "/standard/sds_baek_energetic/HMX_mesoscale_9pt5_GPa/preprocessed/train",  # MESO HIGH RES
]
MIN_MAX_OUTPUT_FILE = f"{BASE_PATH}/LatentPARC/latent_parc/PARCtorch/PARCtorch/data/hmx_min_max.json"
AE_WEIGHTS_PATH     = "autoencoder/weights/basicAE_3000.pth"
SAVE_PATH           = f"{BASE_PATH}/LatentPARC/latent_parc/PARCtorch/PARCtorch/LatentPARC/weights"
WEIGHTS_NAME        = "single_void_basicAE_rollout5_bs1_NoScheduler_lr1e5_3001_FINETUNEunfrozenAE_plateauL1loss_1_1_1" # save name

# ── Flags ─────────────────────────────────────────────────────────────────────
FREEZE_AE           = False       # Freeze autoencoder weights during training
LOAD_WEIGHTS        = True      # Load checkpoint to resume training
LOAD_WEIGHTS_PATH   = "weights/single_void_basicAE_rollout5_bs1_NoScheduler_lr1e5_3001.pth" # Path to checkpoint (only used if LOAD_WEIGHTS=True)


# =============================================================================
# DATA
# =============================================================================

# Compute and save min/max normalization stats
compute_min_max(DATA_DIRS, MIN_MAX_OUTPUT_FILE)

min_max_path = os.path.join(path, "data", "hmx_min_max.json")

# Build dataset
train_dataset = GenericPhysicsDataset(
    data_dirs=[DATA_DIRS[0]],
    future_steps=N_TIMESTEPS - 1,
    min_max_path=min_max_path,
)

# Train / validation split
validation_size = int(len(train_dataset) * VALIDATION_SPLIT)
train_size = len(train_dataset) - validation_size
train_subset, val_subset = random_split(train_dataset, [train_size, validation_size])

# DataLoaders
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=custom_collate_fn,
)
val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=custom_collate_fn,
)


# =============================================================================
# MODEL
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder + dynamics components
encoder       = Encoder(layers=LAYER_SIZES, latent_dim=LATENT_DIM).to(device)
decoder       = Decoder(layers=LAYER_SIZES, latent_dim=LATENT_DIM).to(device)
differentiator = Differentiator(latent_dim=LATENT_DIM)
integrator    = RK4().to(device)

# Load pretrained autoencoder weights
ckpt = torch.load(AE_WEIGHTS_PATH, map_location=device, weights_only=False)
encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("encoder.")})
decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("decoder.")})

# Optionally freeze AE weights
if FREEZE_AE:
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

# Assemble full LatentPARC model
model_init = lp_model(encoder, decoder, differentiator, integrator).to(device)


# =============================================================================
# OPTIMIZER
# =============================================================================

optimizer = Adam(model_init.parameters(), lr=LEARNING_RATE)


# =============================================================================
# RESUME FROM CHECKPOINT (optional)
# =============================================================================

if LOAD_WEIGHTS:
    ckpt = torch.load(LOAD_WEIGHTS_PATH, map_location=device, weights_only=False)
    model_init.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if FREEZE_AE:
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False

# =============================================================================
# SCHEDULER
# =============================================================================
if scheduler == "step":
    scheduler = StepLR(
        optimizer,
        step_size=200,  # how many epochs before decay
        gamma=0.5       # multiply LR by this factor
    )
    
if scheduler == "plateau":
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )


# =============================================================================
# TRAINING
# =============================================================================

model = LatentPARC(model_init, optimizer, SAVE_PATH, WEIGHTS_NAME)

log_dict = model.train(
    loss_function=loss_function,
    epochs=EPOCHS,
    save_every_epochs=SAVE_RATE,
    image_size=IMAGE_SIZE,
    n_channels=N_CHANNELS,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=scheduler,
    noise_fn=add_random_noise,
    initial_max_noise=INITIAL_MAX_NOISE,
    n_reduce_factor=N_REDUCE_FACTOR,
    ms_reduce_factor=MS_REDUCE_FACTOR,
    reduce_on=REDUCE_ON,
    loss_weights=LOSS_WEIGHTS,
    mode=MODE,
)
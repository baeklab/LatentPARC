import torch
import torch.nn as nn
import os
import sys
import logging

from autoencoder_backbones import SkipConSuperResAutoencoder, SuperResHeadAutoencoder
from torch.optim import Adam
from utils import add_random_noise, plot_reconstructions, save_reconstruction_gif, plot_sr_losses, plot_three_reconstructions, save_reconstructions_three_gif
from train_scripts import train_SR_head_autoencoder
from loss_funcs import PerceptualLoss, FFTLoss

import matplotlib.pyplot as plt

base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(base_path)

from data.normalization import compute_min_max

# data_dir_train = "/project/vil_baek/data/physics/PARCTorch/HMX/train"
data_dir_train = "/standard/sds_baek_energetic/data/physics/Shahab_Latent_PARC/Processed_data/PARCtorch_meso_scale/train"

n_ts = 2
min_max_path = os.path.join(base_path, "data", "hmx_min_max.json")
batch_size = 1
validation_split = 0.05

data_dirs = [data_dir_train]
output_file = os.path.join(base_path, "data", "hmx_min_max.json")
compute_min_max(data_dirs, output_file)

from torch.utils.data import DataLoader, random_split
from data.dataset import GenericPhysicsDataset, custom_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

train_dataset = GenericPhysicsDataset(
    data_dirs=data_dirs,
    future_steps=n_ts - 1,
    min_max_path=min_max_path,
)

validation_size = int(len(train_dataset) * validation_split)
train_size = len(train_dataset) - validation_size
train_subset, val_subset = random_split(train_dataset, [train_size, validation_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                          num_workers=1, pin_memory=True, collate_fn=custom_collate_fn)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                          num_workers=1, pin_memory=True, collate_fn=custom_collate_fn)

# ------------------------
# Training Hyperparameters
# ------------------------
save_path    = os.path.join(base_path, "LatentPARC", "autoencoder", "weights")
weights_name = "SRH_frozenBasicAE3000onsingleEMdata_SigOut_FFTLoss_SRHfinetune_onShahabData" # if exists, will continue training from checkpoint, else will create
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load in pretrained encoder/decoder and freeze encoder/decoder options
freeze_backbone = True  # <-- toggle this
pretrain_weights = "basicAE_3000.pth" # optional, if don't want, comment out

epochs            = 1000
save_every_epochs = 100
image_size        = [160, 240] #[128, 256]
n_channels        = 3

initial_max_noise = 0.0 # 0.16
n_reduce_factor   = 0.8
reduce_noise_on   = 500

layer_sizes  = [3, 8]
latent_dim   = 8
sharp_weight  = 1.0
blurry_weight = 0.0 # 0.5
out_act = nn.Sigmoid() # final activation function of the SR head, options None or any activation function

# loss_function = torch.nn.L1Loss().to(device)
# loss_function = PerceptualLoss(
#     pixel_loss=nn.L1Loss(),
#     lambda_pixel=1.0,
#     lambda_perceptual=0.01,
#     device=device
# )
loss_function = FFTLoss(
    pixel_loss=nn.L1Loss(),
    lambda_pixel=1.0,
    lambda_fft=0.1
)

# ------------------------
# Model setup
# ------------------------
weights_path = os.path.join(save_path, f"{weights_name}.pth")

# can choose between SkipConSuperResAutoencoder and SuperResHeadAutoencoder
model = SuperResHeadAutoencoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU(), out_act=out_act).to(device)

criterion = torch.nn.L1Loss().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = None

# ------------------------
# Optionally load from checkpoint
# ------------------------
if os.path.exists(weights_path):
    print(f"Loading SR checkpoint from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    initial_max_noise = checkpoint.get("current_max_noise", initial_max_noise)
else:
    # Check for an old plain AE checkpoint to warm-start encoder/decoder weights
    # TODO: doesn't work with skip con SRH because of encoder mismatch
    old_ae_path = os.path.join(save_path, pretrain_weights)
    if os.path.exists(old_ae_path):
        print(f"No SR checkpoint found. Warm-starting from {old_ae_path}")
        old_checkpoint = torch.load(old_ae_path, map_location=device)

        missing, unexpected = model.load_state_dict(
            old_checkpoint["model_state_dict"], strict=False
        )
        print(f"  {len(missing)} missing keys (SR head), {len(unexpected)} unexpected keys")

        # ---- Freeze encoder/decoder if requested ----
        if freeze_backbone:
            print("Freezing encoder and decoder weights")

            # assumes model.encoder and model.decoder exist
            for param in model.encoder.parameters():
                param.requires_grad = False

            for param in model.decoder.parameters():
                param.requires_grad = False
            # re-define optimizer last to account for frozen modules 
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3
            )
    else:
        print("No checkpoint found, starting from scratch.")

# ------------------------
# Train
# ------------------------
log_dict = train_SR_head_autoencoder(
    model,
    optimizer,
    loss_function,
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
    weights_name=weights_name,
    sharp_weight=sharp_weight,
    blurry_weight=blurry_weight,
)
#!/usr/bin/env python3
"""
Evaluation Script for Autoencoder Metric Computation (Metrics Only)

This script loads a trained Autoencoder checkpoint, prepares an evaluation DataLoader,
runs a forward pass through the model, computes evaluation metrics such as
RMSE for each channel, and calculates hotspot metrics (temperature, area, and their
rates of change).

NOTE on Autoencoder evaluation:
    The Autoencoder reconstructs its *input* rather than predicting future timesteps.
    Each forward pass takes a single frame (or small window) and outputs a reconstruction
    of that same frame.  Accordingly:
      - ic  : the input frame(s) fed to the model  (shape: [B, C, H, W])
      - pred: the reconstructed output              (shape: [B, C, H, W])
      - gt  : the same input frame used as target   (shape: [B, C, H, W])
    All per-channel RMSE and hotspot metrics therefore measure reconstruction quality.

Usage:
    python ae_quantitative_metrics.py \
        --model_checkpoint <path_to_checkpoint> \
        --eval_data        <evaluation_data_directory> \
        --min_max_path     <path_to_min_max_file> \
        [--batch_size <batch_size>] \
        [--device <cuda/cpu>]

Arguments:
    --model_checkpoint : Path to the saved Autoencoder checkpoint (.pth).
    --eval_data        : Directory containing evaluation data.
    --min_max_path     : Path to the min-max normalisation file (JSON).
    --batch_size       : Batch size for evaluation (default: 1).
    --device           : Device to use ("cuda" or "cpu", default: "cuda").

Author: (adapted from joseph_quantitative_metrics.py)
"""

import os
import torch
import torch.nn as nn
import argparse
import logging
import sys
import json

import numpy as np
import scipy.stats as st
from math import sqrt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

# ---------------------------
# Add project root to system path
# ---------------------------
path = os.path.abspath(os.path.join(os.getcwd(), ".."))
print(path)
sys.path.append(path)

# ---------------------------
# Imports — mirrors trainAE.py
# ---------------------------
from data.dataset import GenericPhysicsDataset, custom_collate_fn
from autoencoder_backbones import Encoder, Decoder, Autoencoder


# ---------------------------
# EmLoss — hotspot metrics
# ---------------------------
class EmLoss:
    def __init__(self, cell_area=(1.5 / 128) * (3 / 256), threshold=875, dt=0.17, **kwargs):
        super(EmLoss, self).__init__(**kwargs)
        self.cell_area = cell_area
        self.threshold = threshold
        self.dt = dt

    def compute_KLD(self, y_true, y_pred):
        mean_X = np.mean(y_true)
        sigma_X = np.std(y_true)
        mean_Y = np.mean(y_pred)
        sigma_Y = np.std(y_pred)
        v1 = sigma_X ** 2
        v2 = sigma_Y ** 2
        a = np.log(sigma_Y / sigma_X)
        num = v1 + (mean_X - mean_Y) ** 2
        den = 2 * v2
        b = num / den
        return a + b - 0.5

    def compute_quantitative_evaluation_sensitivity(self, y_trues, y_preds):
        pcc_list, rmse_list, kld_list = [], [], []
        ts = y_preds.shape[1]
        for i in range(ts):
            pcc = st.pearsonr(y_trues[:, i], y_preds[:, i])[0]
            temp_rmse = sqrt(mean_squared_error(y_trues[:, i], y_preds[:, i]))
            kld = self.compute_KLD(y_trues[:, i], y_preds[:, i])
            pcc_list.append(pcc)
            rmse_list.append(temp_rmse)
            kld_list.append(kld)
        return np.mean(rmse_list), np.mean(kld_list), np.mean(pcc_list)

    def _calculate_hotspot_metric(self, Ts, n_timesteps):
        """
        Ts: numpy array of shape (H, W, T) — spatial dims first, time last.
        """
        A_hs_list, T_hs_list = [], []
        for t in range(n_timesteps):
            temp_t = Ts[:, :, t]
            hotspot_mask = (temp_t >= self.threshold).astype(np.float32)
            A_hs = np.sum(hotspot_mask) * self.cell_area
            A_hs_list.append(A_hs)
            if A_hs > 0:
                T_hs = np.sum(temp_t * hotspot_mask * self.cell_area) / A_hs
            else:
                T_hs = 0.0
            T_hs_list.append(T_hs)
        return A_hs_list, T_hs_list

    def calculate_hotspot_metric(self, T_cases, cases_range, n_timesteps):
        all_A_hs, all_T_hs = [], []
        for i in range(cases_range[0], cases_range[1]):
            A_hs, T_hs = self._calculate_hotspot_metric(T_cases[i], n_timesteps)
            all_A_hs.append(A_hs)
            all_T_hs.append(T_hs)
        all_A_hs = np.array(all_A_hs)
        all_T_hs = np.array(all_T_hs)
        mean_T_hs = np.mean(all_T_hs, axis=0)
        mean_A_hs = np.mean(all_A_hs, axis=0)
        perc95_T = np.percentile(all_T_hs, 95, axis=0)
        perc5_T = np.percentile(all_T_hs, 5, axis=0)
        perc95_A = np.percentile(all_A_hs, 95, axis=0)
        perc5_A = np.percentile(all_A_hs, 5, axis=0)
        return (mean_T_hs, perc95_T, perc5_T, all_T_hs), (mean_A_hs, perc95_A, perc5_A, all_A_hs)

    def calculate_hotspot_metric_rate_of_change(self, T_cases, cases_range, n_timesteps):
        hs_temp, hs_area = self.calculate_hotspot_metric(T_cases, cases_range, n_timesteps)
        all_T_hs, all_A_hs = hs_temp[3], hs_area[3]
        rate_T = (all_T_hs[:, 1:] - all_T_hs[:, :-1]) / self.dt
        rate_A = (all_A_hs[:, 1:] - all_A_hs[:, :-1]) / self.dt
        mean_rate_T = np.mean(rate_T, axis=0)
        mean_rate_A = np.mean(rate_A, axis=0)
        perc95_rate_T = np.percentile(rate_T, 95, axis=0)
        perc5_rate_T = np.percentile(rate_T, 5, axis=0)
        perc95_rate_A = np.percentile(rate_A, 95, axis=0)
        perc5_rate_A = np.percentile(rate_A, 5, axis=0)
        return (mean_rate_T, perc95_rate_T, perc5_rate_T, rate_T), \
               (mean_rate_A, perc95_rate_A, perc5_rate_A, rate_A)


# ---------------------------
# Build Autoencoder model
# ---------------------------
def build_model(device):
    """
    Reconstruct the Autoencoder architecture used in trainAE.py.
    Adjust layer_sizes / latent_dim if you changed them during training.
    """
    layer_sizes = [3, 8]
    latent_dim = 8

    encoder = Encoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    decoder = Decoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    model = Autoencoder(encoder, decoder).to(device)
    return model


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Script for Autoencoder (Metrics Only)"
    )
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the Autoencoder checkpoint (.pth)")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Directory containing evaluation data")
    parser.add_argument("--min_max_path", type=str, required=True,
                        help="Path to the min-max normalisation JSON file")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (default: 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help='Device: "cuda" or "cpu" (default: "cuda")')
    args = parser.parse_args()

    device = args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {device}")

    # ---- Build and load model ----
    model = build_model(device)
    logging.info("Autoencoder architecture built.")

    checkpoint = torch.load(args.model_checkpoint, map_location=device)

    # trainAE.py saves a dict with "model_state_dict"; fall back to raw state dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logging.info("Model weights loaded and set to evaluation mode.")

    # ---- Dataset / DataLoader ----
    eval_dataset = GenericPhysicsDataset(
        data_dirs=[args.eval_data],
        future_steps=14,         
        min_max_path=args.min_max_path,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=(device == "cuda"),
        collate_fn=custom_collate_fn,
    )
    logging.info(f"Total evaluation samples: {len(eval_dataset)}")

    # ---- Normalisation helpers ----
    with open(args.min_max_path, "r") as f:
        norm_params = json.load(f)
    print("Normalisation keys:", list(norm_params.keys()))

    json_key_mapping = {
        "Temperature (T)": 0,
        "Pressure (P)":    1,
    }

    def denormalize(channel_name, tensor):
        idx = json_key_mapping[channel_name]
        ch_min = norm_params["channel_min"][idx]
        ch_max = norm_params["channel_max"][idx]
        return tensor * (ch_max - ch_min) + ch_min

    # --------------------------------------------------
    # 1.  Per-channel RMSE  (reconstruction quality)
    # --------------------------------------------------
    # The AE receives ic (shape [B, C, H, W]) and outputs a reconstruction of
    # the same frame.  Ground truth = ic itself (denormalised).
    rmse_channels = {
        "Temperature (T)": {"sse": 0.0, "count": 0},
        "Pressure (P)":    {"sse": 0.0, "count": 0},
    }
    channel_indices = {
        "Temperature (T)": 0,
        "Pressure (P)":    1,
    }

    with torch.no_grad():
        for batch in eval_loader:
            # ic, t0, t1, _ground_truth = batch

            # # Keep only the first 3 channels expected by the encoder
            # ic = ic[:, :3, ...].to(device)   # [B, 3, H, W]

            # # Forward pass through the autoencoder
            # reconstruction = model(ic)        # [B, 3, H, W]

            ic, _, _, target = batch
            ic = ic[:n_channels, ...]
            target = target[:, :3, ...]
            all_test_images = torch.concat((ic.unsqueeze(0), target))
            
            # Initialize list to store all reconstructed images
            all_reconstructed_images = []
            
            for i in range(15):
                with torch.no_grad():
                    img = all_test_images[i, ...].unsqueeze(0).to(device)  # Move image to same device
                    reconstructed_img = model.network(img)
                all_reconstructed_images.append(reconstructed_img.cpu())  # move output back to CPU
            
            # Concatenate all batches to create single tensors
            reconstruction = torch.cat(all_reconstructed_images, dim=0)

            for channel_name, ch in channel_indices.items():
                pred_denorm = denormalize(channel_name, reconstruction[:, ch, :, :])
                gt_denorm   = denormalize(channel_name, ic[:, ch, :, :])
                diff = pred_denorm - gt_denorm
                rmse_channels[channel_name]["sse"]   += diff.pow(2).sum().item()
                rmse_channels[channel_name]["count"] += diff.numel()

    for channel_name, metrics in rmse_channels.items():
        rmse_value = (metrics["sse"] / metrics["count"]) ** 0.5
        logging.info(f"{channel_name} RMSE: {rmse_value:.6f}")
        print(f"{channel_name} RMSE: {rmse_value:.6f}")

    # --------------------------------------------------
    # 2.  Hotspot metrics  (Temperature channel only)
    # --------------------------------------------------
    # We accumulate per-sample spatial temperature maps across the dataset.
    # Each "case" here is one sample (batch element); for batch_size > 1 we
    # split them.  The time axis is a single step (t=0 only), so rate-of-
    # change metrics are computed between the input frame and its reconstruction,
    # interpreted as two "timesteps" separated by dt.
    #
    # If your dataset already returns sequences of frames (future_steps > 1)
    # you can loop over them below and concatenate along the time axis.

    pred_temp_cases_list = []
    gt_temp_cases_list   = []

    with torch.no_grad():
        for batch in eval_loader:
            ic, t0, t1, _ground_truth = batch
            ic = ic[:, :3, ...].to(device)        # [B, 3, H, W]
            reconstruction = model(ic)             # [B, 3, H, W]

            B = ic.shape[0]

            # Temperature channel (index 0), denormalised
            pred_temp = denormalize("Temperature (T)", reconstruction[:, 0, :, :])  # [B, H, W]
            gt_temp   = denormalize("Temperature (T)", ic[:, 0, :, :])              # [B, H, W]

            pred_temp_np = pred_temp.cpu().numpy()  # [B, H, W]
            gt_temp_np   = gt_temp.cpu().numpy()

            # EmLoss expects cases of shape (H, W, T), so we add a singleton T dim
            # and split along the batch axis so each element is one case.
            for b in range(B):
                # shape: (H, W, 1)
                pred_temp_cases_list.append(pred_temp_np[b, :, :, np.newaxis])
                gt_temp_cases_list.append(gt_temp_np[b, :, :, np.newaxis])

    # Stack into (N_cases, H, W, T)
    pred_temp_cases_all = np.stack(pred_temp_cases_list, axis=0)  # (N, H, W, 1)
    gt_temp_cases_all   = np.stack(gt_temp_cases_list,   axis=0)

    n_cases     = pred_temp_cases_all.shape[0]
    n_timesteps = pred_temp_cases_all.shape[-1]   # 1 here

    em_loss = EmLoss()  # threshold=875 K, dt=0.17 s

    pred_hs_temp, pred_hs_area = em_loss.calculate_hotspot_metric(
        pred_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
    )
    gt_hs_temp, gt_hs_area = em_loss.calculate_hotspot_metric(
        gt_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
    )

    rmse_T_hs = np.sqrt(np.mean((np.array(pred_hs_temp[0]) - np.array(gt_hs_temp[0])) ** 2))
    rmse_A_hs = np.sqrt(np.mean((np.array(pred_hs_area[0]) - np.array(gt_hs_area[0])) ** 2))

    logging.info(f"Hotspot Temperature RMSE (T_hs): {rmse_T_hs:.6f}")
    logging.info(f"Hotspot Area RMSE       (A_hs): {rmse_A_hs:.6f}")
    print(f"Hotspot Temperature RMSE (T_hs): {rmse_T_hs:.6f}")
    print(f"Hotspot Area RMSE       (A_hs): {rmse_A_hs:.6f}")

    # Rate-of-change metrics only make sense with ≥2 timesteps.
    if n_timesteps >= 2:
        pred_rate_hs_temp, pred_rate_hs_area = em_loss.calculate_hotspot_metric_rate_of_change(
            pred_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
        )
        gt_rate_hs_temp, gt_rate_hs_area = em_loss.calculate_hotspot_metric_rate_of_change(
            gt_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
        )
        rmse_dotT_hs = np.sqrt(
            np.mean((np.array(pred_rate_hs_temp[0]) - np.array(gt_rate_hs_temp[0])) ** 2)
        )
        rmse_dotA_hs = np.sqrt(
            np.mean((np.array(pred_rate_hs_area[0]) - np.array(gt_rate_hs_area[0])) ** 2)
        )
        logging.info(f"Hotspot Temp Rate-of-Change RMSE (dotT_hs): {rmse_dotT_hs:.6f}")
        logging.info(f"Hotspot Area Rate-of-Change RMSE (dotA_hs): {rmse_dotA_hs:.6f}")
        print(f"Hotspot Temp Rate-of-Change RMSE (dotT_hs): {rmse_dotT_hs:.6f}")
        print(f"Hotspot Area Rate-of-Change RMSE (dotA_hs): {rmse_dotA_hs:.6f}")
    else:
        logging.info(
            "Skipping rate-of-change hotspot metrics: need ≥2 timesteps "
            "(set future_steps > 1 in GenericPhysicsDataset to enable)."
        )
        print(
            "Skipping rate-of-change hotspot metrics: only 1 timestep available. "
            "Increase future_steps in the dataset to enable."
        )

    logging.info("Evaluation completed.")


if __name__ == "__main__":
    main()
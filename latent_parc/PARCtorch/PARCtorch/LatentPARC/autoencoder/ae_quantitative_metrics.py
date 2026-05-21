#!/usr/bin/env python3
"""
Evaluation Script for Autoencoder Metric Computation — Full Sequence (Metrics Only)

Each frame in the sequence is passed independently through the Autoencoder
(encode → decode). The reconstruction is compared against the original input
frame at every timestep, so all downstream hotspot and rate-of-change metrics
work exactly as in the original LatentPARC evaluation — the only difference is
that "prediction" here means "reconstruction", not "dynamics forecast".

Usage:
    python ae_quantitative_metrics.py \
        --model_checkpoint <path_to_checkpoint> \
        --eval_data        <evaluation_data_directory> \
        --min_max_path     <path_to_min_max_file> \
        [--batch_size <batch_size>] \
        [--device <cuda/cpu>]
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
path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
print(path)
sys.path.append(path)

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
        mean_X, sigma_X = np.mean(y_true), np.std(y_true)
        mean_Y, sigma_Y = np.mean(y_pred), np.std(y_pred)
        v1, v2 = sigma_X ** 2, sigma_Y ** 2
        a = np.log(sigma_Y / sigma_X)
        b = (v1 + (mean_X - mean_Y) ** 2) / (2 * v2)
        return a + b - 0.5

    def compute_quantitative_evaluation_sensitivity(self, y_trues, y_preds):
        pcc_list, rmse_list, kld_list = [], [], []
        for i in range(y_preds.shape[1]):
            pcc_list.append(st.pearsonr(y_trues[:, i], y_preds[:, i])[0])
            rmse_list.append(sqrt(mean_squared_error(y_trues[:, i], y_preds[:, i])))
            kld_list.append(self.compute_KLD(y_trues[:, i], y_preds[:, i]))
        return np.mean(rmse_list), np.mean(kld_list), np.mean(pcc_list)

    def _calculate_hotspot_metric(self, Ts, n_timesteps):
        """Ts: (H, W, T)"""
        A_hs_list, T_hs_list = [], []
        for t in range(n_timesteps):
            temp_t = Ts[:, :, t]
            hotspot_mask = (temp_t >= self.threshold).astype(np.float32)
            A_hs = np.sum(hotspot_mask) * self.cell_area
            A_hs_list.append(A_hs)
            T_hs = (np.sum(temp_t * hotspot_mask * self.cell_area) / A_hs) if A_hs > 0 else 0.0
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
        return (
            (np.mean(all_T_hs, axis=0), np.percentile(all_T_hs, 95, axis=0),
             np.percentile(all_T_hs, 5, axis=0), all_T_hs),
            (np.mean(all_A_hs, axis=0), np.percentile(all_A_hs, 95, axis=0),
             np.percentile(all_A_hs, 5, axis=0), all_A_hs),
        )

    def calculate_hotspot_metric_rate_of_change(self, T_cases, cases_range, n_timesteps):
        hs_temp, hs_area = self.calculate_hotspot_metric(T_cases, cases_range, n_timesteps)
        all_T_hs, all_A_hs = hs_temp[3], hs_area[3]
        rate_T = (all_T_hs[:, 1:] - all_T_hs[:, :-1]) / self.dt
        rate_A = (all_A_hs[:, 1:] - all_A_hs[:, :-1]) / self.dt
        return (
            (np.mean(rate_T, axis=0), np.percentile(rate_T, 95, axis=0),
             np.percentile(rate_T, 5, axis=0), rate_T),
            (np.mean(rate_A, axis=0), np.percentile(rate_A, 95, axis=0),
             np.percentile(rate_A, 5, axis=0), rate_A),
        )


# ---------------------------
# Build Autoencoder model — matches trainAE.py
# ---------------------------
def build_model(device):
    layer_sizes = [3, 8]
    latent_dim  = 8
    encoder = Encoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    decoder = Decoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    return Autoencoder(encoder, decoder).to(device)


# ---------------------------
# Reconstruct a full sequence frame-by-frame
# ---------------------------
def reconstruct_sequence(model, sequence, device):
    """
    Args:
        sequence : Tensor (T, B, C, H, W) — full normalised sequence
        model    : trained Autoencoder

    Returns:
        Tensor (T, B, C, H, W) — per-frame reconstructions, on CPU
    """
    reconstructions = []
    for t in range(sequence.shape[0]):
        frame = sequence[t].to(device)          # (B, C, H, W)
        reconstructions.append(model(frame).cpu())
    return torch.stack(reconstructions, dim=0)  # (T, B, C, H, W)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Autoencoder evaluation — full sequence reconstruction metrics"
    )
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--eval_data",         type=str, required=True)
    parser.add_argument("--min_max_path",      type=str, required=True)
    parser.add_argument("--batch_size",        type=int, default=1)
    parser.add_argument("--device",            type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {device}")

    # ---- Model ----
    model = build_model(device)
    logging.info("Autoencoder architecture built.")

    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    logging.info("Weights loaded, model in eval mode.")

    # ---- Dataset ----
    # future_steps=14 → dataset returns ic (frame 0) + 14 future frames = 15 total
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

    json_key_mapping = {"Temperature (T)": 0, "Pressure (P)": 1}

    def denormalize(channel_name, tensor):
        idx    = json_key_mapping[channel_name]
        ch_min = norm_params["channel_min"][idx]
        ch_max = norm_params["channel_max"][idx]
        return tensor * (ch_max - ch_min) + ch_min

    # -------------------------------------------------------
    # Single unified loop — RMSE accumulation + hotspot data
    # -------------------------------------------------------
    # GenericPhysicsDataset returns:
    #   ic           : (B, C, H, W)      — frame 0
    #   ground_truth : (14, B, C, H, W)  — frames 1..14
    #
    # We prepend ic to get a full (15, B, C, H, W) sequence, pass every
    # frame independently through the AE, and treat the reconstruction as
    # the "prediction". Ground truth = the original frames.

    channel_indices = {"Temperature (T)": 0, "Pressure (P)": 1}
    rmse_channels   = {k: {"sse": 0.0, "count": 0} for k in channel_indices}

    pred_temp_cases_list = []
    gt_temp_cases_list   = []

    with torch.no_grad():
        for batch in eval_loader:
            ic, t0, t1, ground_truth = batch

            # ic:           (B, C,  H, W)
            # ground_truth: (14, B, C, H, W)
            ic           = ic[:, :3, ...]                       # keep first 3 channels
            ground_truth = ground_truth[:, :, :3, ...]

            # Full 15-frame sequence on CPU until reconstruct_sequence moves it
            full_sequence = torch.cat(
                [ic.unsqueeze(0), ground_truth], dim=0
            )                                                   # (15, B, 3, H, W)

            # Reconstruct every frame independently, result back on CPU
            reconstructions = reconstruct_sequence(model, full_sequence, device)
            # reconstructions: (15, B, 3, H, W)

            # ---- per-channel RMSE ----
            for channel_name, ch in channel_indices.items():
                pred_denorm = denormalize(channel_name, reconstructions[:, :, ch, :, :])
                gt_denorm   = denormalize(channel_name, full_sequence[:, :, ch, :, :])
                diff = pred_denorm - gt_denorm
                rmse_channels[channel_name]["sse"]   += diff.pow(2).sum().item()
                rmse_channels[channel_name]["count"] += diff.numel()

            # ---- temperature maps for hotspot metrics ----
            # (15, B, H, W) → (B, H, W, 15) so each batch element is one case
            pred_temp_np = denormalize(
                "Temperature (T)", reconstructions[:, :, 0, :, :]
            ).numpy()
            gt_temp_np = denormalize(
                "Temperature (T)", full_sequence[:, :, 0, :, :]
            ).numpy()

            pred_temp_np = np.transpose(pred_temp_np, (1, 2, 3, 0))  # (B, H, W, T)
            gt_temp_np   = np.transpose(gt_temp_np,   (1, 2, 3, 0))

            for b in range(pred_temp_np.shape[0]):
                pred_temp_cases_list.append(pred_temp_np[b])   # (H, W, T)
                gt_temp_cases_list.append(gt_temp_np[b])

    # ---- Print per-channel RMSE ----
    for channel_name, metrics in rmse_channels.items():
        rmse_value = (metrics["sse"] / metrics["count"]) ** 0.5
        logging.info(f"{channel_name} RMSE: {rmse_value:.6f}")
        print(f"{channel_name} RMSE: {rmse_value:.6f}")

    # ---- Hotspot metrics ----
    pred_temp_cases_all = np.stack(pred_temp_cases_list, axis=0)  # (N, H, W, T)
    gt_temp_cases_all   = np.stack(gt_temp_cases_list,   axis=0)

    n_cases     = pred_temp_cases_all.shape[0]
    n_timesteps = pred_temp_cases_all.shape[-1]                   # 15
    logging.info(f"Hotspot metric cases: {n_cases}, timesteps: {n_timesteps}")

    em_loss = EmLoss()

    pred_hs_temp, pred_hs_area = em_loss.calculate_hotspot_metric(
        pred_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
    )
    gt_hs_temp, gt_hs_area = em_loss.calculate_hotspot_metric(
        gt_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
    )

    rmse_T_hs = np.sqrt(np.mean((pred_hs_temp[0] - gt_hs_temp[0]) ** 2))
    rmse_A_hs = np.sqrt(np.mean((pred_hs_area[0] - gt_hs_area[0]) ** 2))

    logging.info(f"Hotspot Temperature RMSE (T_hs): {rmse_T_hs:.6f}")
    logging.info(f"Hotspot Area RMSE       (A_hs): {rmse_A_hs:.6f}")
    print(f"Hotspot Temperature RMSE (T_hs): {rmse_T_hs:.6f}")
    print(f"Hotspot Area RMSE       (A_hs): {rmse_A_hs:.6f}")

    # Rate-of-change (always valid with 15 timesteps)
    pred_rate_hs_temp, pred_rate_hs_area = em_loss.calculate_hotspot_metric_rate_of_change(
        pred_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
    )
    gt_rate_hs_temp, gt_rate_hs_area = em_loss.calculate_hotspot_metric_rate_of_change(
        gt_temp_cases_all, (0, n_cases), n_timesteps=n_timesteps
    )

    rmse_dotT_hs = np.sqrt(np.mean((pred_rate_hs_temp[0] - gt_rate_hs_temp[0]) ** 2))
    rmse_dotA_hs = np.sqrt(np.mean((pred_rate_hs_area[0] - gt_rate_hs_area[0]) ** 2))

    logging.info(f"Hotspot Temp Rate-of-Change RMSE (dotT_hs): {rmse_dotT_hs:.6f}")
    logging.info(f"Hotspot Area Rate-of-Change RMSE (dotA_hs): {rmse_dotA_hs:.6f}")
    print(f"Hotspot Temp Rate-of-Change RMSE (dotT_hs): {rmse_dotT_hs:.6f}")
    print(f"Hotspot Area Rate-of-Change RMSE (dotA_hs): {rmse_dotA_hs:.6f}")

    logging.info("Evaluation completed.")


if __name__ == "__main__":
    main()

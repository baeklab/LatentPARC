#!/usr/bin/env python3
"""
Evaluation Script for PARCTorch Metric Computation (Metrics Only)

This script loads a trained model checkpoint, prepares an evaluation DataLoader,
runs a forward pass through the model, computes evaluation metrics such as
RMSE for each channel, and calculates hotspot metrics (temperature, area, and their rates of change)
using the original (old) method. The metrics are printed to the console.

Usage:
    python quantitative_analysis_EM.py --model_checkpoint <path_to_checkpoint> --eval_data <evaluation_data_directory> --min_max_path <path_to_min_max_file> [--batch_size <batch_size>] [--device <cuda/cpu>]

Arguments:
    --model_checkpoint: Path to model checkpoint.
    --eval_data: Directory containing evaluation data.
    --min_max_path: Path to the min-max normalization file.
    --batch_size: Batch size for evaluation (default: 1).
    --device: Device to use for evaluation ("cuda" or "cpu", default: "cuda").

Author: Your Name
Date: 2024-10-10
"""

import os
import torch
import argparse
import logging
from torch.utils.data import DataLoader
import sys
import json

# Additional imports for evaluation
import numpy as np
import scipy.stats as st
from math import sqrt
from sklearn.metrics import mean_squared_error

BASE_PATH = "/sfs/gpfs/tardis/home/pdy2bw/Research/LatentPARC_new" # path to your LatentPARC directory folder

sys.path.append(f"{BASE_PATH}/LatentPARC/latent_parc/PARCtorch")
sys.path.append(f"{BASE_PATH}/LatentPARC/latent_parc/PARCtorch/PARCtorch/LatentPARC")

path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(path)

# ---------------------------
# Import required modules from your project
# ---------------------------

from autoencoder.autoencoder_backbones import *
from autoencoder.utils import add_random_noise
from autoencoder.loss_funcs import FFTLoss
from PARCv1.differentiator import *
from PARCtorch.integrator.rk4 import *
from PARCtorch.integrator.numintegrator import *
from model import *
from utils import *
from data.normalization import compute_min_max
from data.dataset import GenericPhysicsDataset, custom_collate_fn
from utilities.viz import visualize_channels, save_gifs_with_ground_truth


# ---------------------------
# EmLoss Class Integration (Hotspot metrics per literature)
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
        return (mean_rate_T, perc95_rate_T, perc5_rate_T, rate_T), (mean_rate_A, perc95_rate_A, perc5_rate_A, rate_A)

# ---------------------------
# End of EmLoss Class
# ---------------------------

def build_model(device):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define layer sizes and initialize all elements
    layer_sizes = [3, 8]
    latent_dim = 8
    encoder = Encoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    decoder = Decoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    differentiator = Differentiator(latent_dim=latent_dim)
    integrator = RK4().to(device)  # step size may be hyper-param of interest

    # Initialize LatentPARC
    model = lp_model(encoder, decoder, differentiator, integrator).to(device)
    return model

def build_new_model(device, MODEL_VERSION):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    layer_sizes = [3, 8]
    latent_dim = 8
    
    # Map model version to (architecture class, out_act)
    MODEL_CONFIG = {
        "NoSR":    (SuperResHeadAutoencoder,      None),
        "SR":      (SuperResHeadAutoencoder,      nn.Sigmoid()),
        "PhongSR": (SuperResHeadAutoencoderPhong, nn.Sigmoid()),
    }
    
    ae_class, out_act = MODEL_CONFIG[MODEL_VERSION]
    
    # Build and load autoencoder
    sr_ae = ae_class(
        layers=layer_sizes,
        latent_dim=latent_dim,
        act_fn=nn.ReLU(),
        out_act=out_act
    ).to(device)
    
    # Extract submodules
    encoder        = Encoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    decoder        = Decoder(layers=layer_sizes, latent_dim=latent_dim, act_fn=nn.ReLU()).to(device)
    sr_head        = nn.Identity() if MODEL_VERSION == "NoSR" else sr_ae.sr_head
    
    # Dynamics components
    differentiator = Differentiator(latent_dim=latent_dim)
    integrator     = RK4().to(device)
    
    # Assemble full LatentPARC model
    model = lp_model_sr(encoder, decoder, differentiator, integrator, sr_head=sr_head).to(device)
    
    return (model)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_hotspot_metrics(
    gt_hs_temp, gt_hs_area, gt_rate_hs_temp, gt_rate_hs_area,
    pred_hs_temp, pred_hs_area, pred_rate_hs_temp, pred_rate_hs_area,
    dt=0.17, save_path=None
):
    """
    Creates a 2x2 plot of hotspot metrics with mean lines and 5th-95th percentile bands.
    
    Each *_hs_* tuple is structured as: (mean, perc95, perc5, all_values)
    matching the output of calculate_hotspot_metric / calculate_hotspot_metric_rate_of_change.
    """

    # --- Build time axes ---
    n_ts       = len(gt_hs_temp[0])       # number of timesteps
    n_ts_rate  = len(gt_rate_hs_temp[0])  # one fewer (finite differences)

    # Time in ns: starts at 0.68 per your figure, spaced by dt (0.17 ns)
    t_start = 0.68
    t_main = np.array([t_start + i * dt for i in range(n_ts)])
    t_rate = np.array([t_start + i * dt for i in range(n_ts_rate)])

    # --- Color / label config ---
    gt_color   = "black"
    pred_color = "mediumpurple"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    def plot_panel(ax, t, mean_gt, p5_gt, p95_gt, mean_pred, p5_pred, p95_pred,
                   title, ylabel, xlabel="ns"):
        # Ground truth
        ax.plot(t, mean_gt, color=gt_color, linewidth=2, label="Ground Truth")
        ax.fill_between(t, p5_gt, p95_gt, color=gt_color, alpha=0.2)
        # Prediction
        ax.plot(t, mean_pred, color=pred_color, linewidth=2, label="Proposed")
        ax.fill_between(t, p5_pred, p95_pred, color=pred_color, alpha=0.2)

        ax.set_title(title, fontsize=14, loc="left", fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # (a) A_hs
    plot_panel(
        axes[0, 0], t_main,
        gt_hs_area[0],   gt_hs_area[2],   gt_hs_area[1],
        pred_hs_area[0], pred_hs_area[2], pred_hs_area[1],
        title=r"(a)  $A_{hs}$", ylabel=r"$\mu m^2$"
    )

    # (b) A_hs_dot (rate of change)
    plot_panel(
        axes[0, 1], t_rate,
        gt_rate_hs_area[0],   gt_rate_hs_area[2],   gt_rate_hs_area[1],
        pred_rate_hs_area[0], pred_rate_hs_area[2], pred_rate_hs_area[1],
        title=r"(b)  $\dot{A}_{hs}$", ylabel=r"$\mu m^2 / ns$"
    )

    # (c) T_hs
    plot_panel(
        axes[1, 0], t_main,
        gt_hs_temp[0],   gt_hs_temp[2],   gt_hs_temp[1],
        pred_hs_temp[0], pred_hs_temp[2], pred_hs_temp[1],
        title=r"(c)  $T_{hs}$", ylabel=r"$K$"
    )

    # (d) T_hs_dot (rate of change)
    plot_panel(
        axes[1, 1], t_rate,
        gt_rate_hs_temp[0],   gt_rate_hs_temp[2],   gt_rate_hs_temp[1],
        pred_rate_hs_temp[0], pred_rate_hs_temp[2], pred_rate_hs_temp[1],
        title=r"(d)  $\dot{T}_{hs}$", ylabel=r"$K / ns$"
    )

    # --- Shared legend below plots ---
    gt_patch   = mpatches.Patch(color=gt_color,   label="Ground Truth")
    pred_patch = mpatches.Patch(color=pred_color, label="Proposed")
    fig.legend(
        handles=[gt_patch, pred_patch],
        loc="lower center",
        ncol=2,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluation Script for PARCTorch Results (Metrics Only)")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_version", type=str, required=True, help="Options SR, PhongSR, NoSR")
    parser.add_argument("--eval_data", type=str, required=True, help="Directory containing evaluation data")
    parser.add_argument("--min_max_path", type=str, required=True, help="Path to the min-max normalization file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (cuda or cpu)")
    parser.add_argument("--save_results", type=str, default=None,
                    help="Path to save computed metrics as .npz (e.g. results/proposed.npz)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {device}")
    
    model = build_new_model(device, args.model_version)
    logging.info("Model architecture built.")
    
    state_dict = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    logging.info("Model weights loaded and set to evaluation mode.")
    
    eval_dataset = GenericPhysicsDataset(
        data_dirs=[args.eval_data],
        future_steps=14,
        min_max_path=args.min_max_path
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=(device == "cuda"),
        collate_fn=custom_collate_fn
    )
    logging.info(f"Total evaluation samples: {len(eval_dataset)}")
    
    # ---------------------------
    # Compute RMSE for Each Channel
    # ---------------------------
    with open(args.min_max_path, 'r') as f:
        norm_params = json.load(f)      
    print("Normalization keys:", list(norm_params.keys()))
    json_key_mapping = {
        "Temperature (T)": 0,
        "Pressure (P)": 1,
        # "Velocity U (vx)": 3,
        # "Velocity V (vy)": 4,
    }
    def denormalize(channel_name, tensor):
        idx = json_key_mapping.get(channel_name)
        if idx is None:
            raise ValueError(f"Channel {channel_name} not found in mapping.")
        channel_min = norm_params["channel_min"][idx]
        channel_max = norm_params["channel_max"][idx]
        return tensor * (channel_max - channel_min) + channel_min
    
    rmse_channels = {
        "Temperature (T)": {"sse": 0.0, "count": 0},
        "Pressure (P)": {"sse": 0.0, "count": 0},
        # "Velocity U (vx)": {"sse": 0.0, "count": 0},
        # "Velocity V (vy)": {"sse": 0.0, "count": 0},
    }
    channel_indices = {
        "Temperature (T)": 0,
        "Pressure (P)": 1,
        # "Velocity U (vx)": 3,
        # "Velocity V (vy)": 4,
    }
    for batch in eval_loader:
        ic, t0, t1, ground_truth = batch
        ic = ic[:, :3, ...].to(device)
        # t0 = t0.to(device)
        # t1 = t1.to(device)
        n_ts = 15
        ground_truth = ground_truth.to(device)
        ground_truth = ground_truth[4:, ...]
        with torch.no_grad():
            # predictions = model(ic, t0, t1)
            _, predictions = model(ic.to(device), n_ts=n_ts, mode='pred')
            predictions = predictions[5:, ...] # LP model outputs first ts as well
        for channel_name, ch in channel_indices.items():
            pred_denorm = denormalize(channel_name, predictions[:, :, ch, :, :])
            gt_denorm = denormalize(channel_name, ground_truth[:, :, ch, :, :])
            diff = pred_denorm - gt_denorm
            rmse_channels[channel_name]["sse"] += diff.pow(2).sum().item()
            rmse_channels[channel_name]["count"] += diff.numel()
    for channel_name, metrics in rmse_channels.items():
        rmse_value = (metrics["sse"] / metrics["count"]) ** 0.5
        logging.info(f"{channel_name} RMSE: {rmse_value}")
        print(f"{channel_name} RMSE: {rmse_value}")


    # ---------------------------
    # Compute Hotspot Metrics and Their Rates of Change
    # ---------------------------
    # Using the "old" method:
    all_batches = list(eval_loader)
    pred_temp_cases_list = []
    gt_temp_cases_list = []
    for batch in all_batches:
        ic, t0, t1, ground_truth = batch
        ic = ic[:, :3, ...].to(device)
        n_ts = 15
        # t0 = t0.to(device)
        # t1 = t1.to(device)
        ground_truth = ground_truth.to(device)
        ground_truth = ground_truth[4:, ...]
        with torch.no_grad():
            # predictions = model(ic, t0, t1)
            _, predictions = model(ic.to(device), n_ts=n_ts, mode='pred')
            predictions = predictions[5:, ...]
        # Process predictions for Temperature channel (index 0)
        T, B, C, H, W = predictions.shape
        # print(f"Number of timesteps = {T}")
        pred_temp = denormalize("Temperature (T)", predictions[:, :, 0, :, :])
        pred_temp_np = pred_temp.cpu().numpy()
        # Transpose to (timesteps, H, W, batch)
        pred_temp_np = np.transpose(pred_temp_np, (1, 2, 3, 0))
        pred_temp_cases_list.append(pred_temp_np)
        
        gt_temp = denormalize("Temperature (T)", ground_truth[:, :, 0, :, :])
        gt_temp_np = gt_temp.cpu().numpy()
        gt_temp_np = np.transpose(gt_temp_np, (1, 2, 3, 0))
        gt_temp_cases_list.append(gt_temp_np)
        
    pred_temp_cases_all = np.concatenate(pred_temp_cases_list, axis=0)
    gt_temp_cases_all = np.concatenate(gt_temp_cases_list, axis=0)

    em_loss = EmLoss()  # Uses threshold 875 and dt 0.17 by default
    pred_hs_temp, pred_hs_area = em_loss.calculate_hotspot_metric(pred_temp_cases_all, (0, pred_temp_cases_all.shape[0]), n_timesteps=pred_temp_cases_all.shape[-1])
    gt_hs_temp, gt_hs_area = em_loss.calculate_hotspot_metric(gt_temp_cases_all, (0, gt_temp_cases_all.shape[0]), n_timesteps=gt_temp_cases_all.shape[-1])
    rmse_T_hs = np.sqrt(np.mean((np.array(pred_hs_temp[0]) - np.array(gt_hs_temp[0]))**2))
    rmse_A_hs = np.sqrt(np.mean((np.array(pred_hs_area[0]) - np.array(gt_hs_area[0]))**2))
    logging.info(f"Hotspot Temperature RMSE (T_hs): {rmse_T_hs}")
    print(f"Hotspot Temperature RMSE (T_hs): {rmse_T_hs}")
    logging.info(f"Hotspot Area RMSE (A_hs): {rmse_A_hs}")
    print(f"Hotspot Area RMSE (A_hs): {rmse_A_hs}")

    pred_rate_hs_temp, pred_rate_hs_area = em_loss.calculate_hotspot_metric_rate_of_change(pred_temp_cases_all, (0, pred_temp_cases_all.shape[0]), n_timesteps=pred_temp_cases_all.shape[-1])
    gt_rate_hs_temp, gt_rate_hs_area = em_loss.calculate_hotspot_metric_rate_of_change(gt_temp_cases_all, (0, gt_temp_cases_all.shape[0]), n_timesteps=gt_temp_cases_all.shape[-1])
    rmse_dotT_hs = np.sqrt(np.mean((np.array(pred_rate_hs_temp[0]) - np.array(gt_rate_hs_temp[0]))**2))
    rmse_dotA_hs = np.sqrt(np.mean((np.array(pred_rate_hs_area[0]) - np.array(gt_rate_hs_area[0]))**2))
    logging.info(f"Hotspot Temperature Rate of Change RMSE (dotT_hs): {rmse_dotT_hs}")
    logging.info(f"Hotspot Area Rate of Change RMSE (dotA_hs): {rmse_dotA_hs}")
    print(f"Hotspot Temperature Rate of Change RMSE (dotT_hs): {rmse_dotT_hs}")
    print(f"Hotspot Area Rate of Change RMSE (dotA_hs): {rmse_dotA_hs}")

    logging.info("Evaluation completed and metrics printed.")

    # plot_hotspot_metrics(
    #     gt_hs_temp, gt_hs_area, gt_rate_hs_temp, gt_rate_hs_area,
    #     pred_hs_temp, pred_hs_area, pred_rate_hs_temp, pred_rate_hs_area,
    #     dt=em_loss.dt,
    #     save_path="hotspot_metrics.png"   # or None to just show interactively
    # )

    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:  # only call makedirs if there's actually a directory component
            os.makedirs(save_dir, exist_ok=True)
        np.savez(
            args.save_results,
            # --- Ground Truth ---
            # hotspot temp
            mean_T_hs       = gt_hs_temp[0],
            perc95_T_hs     = gt_hs_temp[1],
            perc5_T_hs      = gt_hs_temp[2],
            pred_mean_T_hs  = pred_hs_temp[0],
            pred_perc95_T_hs= pred_hs_temp[1],
            pred_perc5_T_hs = pred_hs_temp[2],
            # hotspot area
            mean_A_hs       = gt_hs_area[0],
            perc95_A_hs     = gt_hs_area[1],
            perc5_A_hs      = gt_hs_area[2],
            pred_mean_A_hs  = pred_hs_area[0],
            pred_perc95_A_hs= pred_hs_area[1],
            pred_perc5_A_hs = pred_hs_area[2],
            # rate of change temp
            mean_dotT_hs        = gt_rate_hs_temp[0],
            perc95_dotT_hs      = gt_rate_hs_temp[1],
            perc5_dotT_hs       = gt_rate_hs_temp[2],
            pred_mean_dotT_hs   = pred_rate_hs_temp[0],
            pred_perc95_dotT_hs = pred_rate_hs_temp[1],
            pred_perc5_dotT_hs  = pred_rate_hs_temp[2],
            # rate of change area
            mean_dotA_hs        = gt_rate_hs_area[0],
            perc95_dotA_hs      = gt_rate_hs_area[1],
            perc5_dotA_hs       = gt_rate_hs_area[2],
            pred_mean_dotA_hs   = pred_rate_hs_area[0],
            pred_perc95_dotA_hs = pred_rate_hs_area[1],
            pred_perc5_dotA_hs  = pred_rate_hs_area[2],
        )
        print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    main()
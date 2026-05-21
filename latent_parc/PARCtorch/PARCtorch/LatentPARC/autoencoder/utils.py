import json
import os
import io
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

def add_random_noise(images, min_val=0.0, max_val=0.1):
    """
    Add random (uniform) noise to the images.

    Parameters:
        images: Tensor of input images.
        min_val: Minimum value of the noise.
        max_val: Maximum value of the noise.

    Returns:
        Noisy images.
    """
    noise = torch.rand_like(images) * (max_val - min_val) + min_val
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)  # Keep pixel values in [0, 1]

def save_model_and_logs(
    model,
    optimizer,
    save_path,
    weights_name,
    log_dict,
    epoch,
    scheduler=None,
    current_max_noise=None,
    config=None,
    noise_config=None,
    extra_state=None,
):
    """
    Save full training checkpoint + logs.
    """

    if not save_path:
        return

    os.makedirs(save_path, exist_ok=True)

    # --- Core checkpoint ---
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # --- Scheduler ---
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # --- Current training state (dynamic things) ---
    if current_max_noise is not None:
        checkpoint["current_max_noise"] = current_max_noise

    # --- Configs (static + semi-static) ---
    if config is not None:
        checkpoint["config"] = config

    if noise_config is not None:
        # Optionally inject current noise for readability
        noise_config = dict(noise_config)  # avoid mutating original
        if current_max_noise is not None:
            noise_config["current_max_noise"] = current_max_noise

        checkpoint["noise_config"] = noise_config

    # --- Extra ---
    if extra_state is not None:
        checkpoint["extra_state"] = extra_state

    # --- Save checkpoint ---
    ckpt_path = os.path.join(save_path, f"{weights_name}_{epoch + 1}.pth")
    torch.save(checkpoint, ckpt_path)

    # --- Save logs ---
    log_path = os.path.join(save_path, f"{weights_name}_{epoch + 1}.json")
    with open(log_path, "w") as f:
        json.dump(log_dict, f, indent=2)

    print(f"Checkpoint + logs saved to {save_path}")
    

def plot_reconstructions(original, reconstructed, title="Original vs Reconstructed",
                         channel_names=None, num_samples=None, cmap="jet", figsize=(12, 9),
                         clamp=False):
    """
    Plot original vs reconstructed images channel by channel.
    Args:
        original:      tensor or array [N, C, H, W]
        reconstructed: tensor or array [N, C, H, W]
        title:         base title for each sample's figure
        channel_names: list of channel name strings, defaults to ["Channel 0", "Channel 1", ...]
        num_samples:   how many samples to plot, defaults to all
        cmap:          matplotlib colormap
        figsize:       figure size per sample
        clamp:         if True, clamp all values to [0, 1] before plotting
    """
    if hasattr(original, 'cpu'):
        original = original.cpu()
    if hasattr(reconstructed, 'cpu'):
        reconstructed = reconstructed.cpu()

    if clamp:
        original      = original.clamp(0, 1) if hasattr(original, 'clamp') else np.clip(original, 0, 1)
        reconstructed = reconstructed.clamp(0, 1) if hasattr(reconstructed, 'clamp') else np.clip(reconstructed, 0, 1)

    n, c, h, w = original.shape
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(c)]
    if num_samples is None:
        num_samples = n
    global_min = min(original.min(), reconstructed.min())
    global_max = max(original.max(), reconstructed.max())
    for sample_idx in range(num_samples):
        fig, axes = plt.subplots(c, 2, figsize=figsize)
        if c == 1:
            axes = axes[None, :]
        fig.suptitle(f"{title} — Sample {sample_idx + 1}", fontsize=16)
        for channel_idx, channel_name in enumerate(channel_names):
            original_channel      = original[sample_idx, channel_idx].numpy()
            reconstructed_channel = reconstructed[sample_idx, channel_idx].numpy()
            ax = axes[channel_idx, 0]
            im = ax.imshow(original_channel, cmap=cmap, aspect="auto",
                           vmin=global_min, vmax=global_max)
            ax.set_title(f"Original — {channel_name}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)
            ax = axes[channel_idx, 1]
            im = ax.imshow(reconstructed_channel, cmap=cmap, aspect="auto",
                           vmin=global_min, vmax=global_max)
            ax.set_title(f"Reconstructed — {channel_name}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_three_reconstructions(original, reconstructed, ground_truth=None, title="Original vs Reconstructed",
                         channel_names=None, num_samples=None, cmap="jet", figsize=None,
                         show_colorbar=True, clamp=False):
    """
    Plot original vs reconstructed (vs optional ground truth) images channel by channel.
    Args:
        original:       tensor or array [N, C, H, W]  (blurry input)
        reconstructed:  tensor or array [N, C, H, W]  (model output / sharp)
        ground_truth:   tensor or array [N, C, H, W]  (optional clean target)
        title:          base title for each sample's figure
        channel_names:  list of channel name strings, defaults to ["Channel 0", "Channel 1", ...]
        num_samples:    how many samples to plot, defaults to all
        cmap:           matplotlib colormap
        figsize:        figure size; if None, auto-sized to preserve aspect ratio
        show_colorbar:  if True, add a colorbar to each subplot
    """
    if hasattr(original, 'cpu'):
        original = original.cpu()
    if hasattr(reconstructed, 'cpu'):
        reconstructed = reconstructed.cpu()
    if ground_truth is not None and hasattr(ground_truth, 'cpu'):
        ground_truth = ground_truth.cpu()

    if clamp:
        original      = original.clamp(0, 1) if hasattr(original, 'clamp') else np.clip(original, 0, 1)
        reconstructed = reconstructed.clamp(0, 1) if hasattr(reconstructed, 'clamp') else np.clip(reconstructed, 0, 1)
        ground_truth  = ground_truth.clamp(0, 1) if hasattr(ground_truth, 'clamp') else np.clip(ground_truth, 0, 1)

    n, c, h, w = original.shape
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(c)]
    if num_samples is None:
        num_samples = n

    columns = [("Blurry", original), ("Sharp", reconstructed)]
    if ground_truth is not None:
        columns.append(("Ground Truth", ground_truth))

    global_min = min(t.min() for _, t in columns)
    global_max = max(t.max() for _, t in columns)
    num_cols = len(columns)

    for sample_idx in range(num_samples):
        if figsize is None:
            aspect_ratio = w / h
            fig, axes = plt.subplots(c, num_cols, figsize=(num_cols * 4 * aspect_ratio, c * 4))
        else:
            fig, axes = plt.subplots(c, num_cols, figsize=figsize)

        if c == 1 and num_cols == 1:
            axes = axes[None, None]
        elif c == 1:
            axes = axes[None, :]
        elif num_cols == 1:
            axes = axes[:, None]

        fig.suptitle(f"{title} — Sample {sample_idx + 1}", fontsize=16)

        for channel_idx, channel_name in enumerate(channel_names):
            for col_idx, (col_label, tensor) in enumerate(columns):
                channel_data = tensor[sample_idx, channel_idx].numpy()
                ax = axes[channel_idx, col_idx]
                im = ax.imshow(channel_data, cmap=cmap, aspect="equal",
                               vmin=global_min, vmax=global_max)
                ax.set_title(f"{col_label} — {channel_name}")
                ax.axis("off")
                if show_colorbar:
                    fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def save_reconstruction_gif(original, reconstructed, file_name, plot_name="Reconstruction",
                             channel_names=None, num_samples=None, cmap="jet",
                             figsize=(6, 4), dpi=150, duration=500):
    """
    Save a GIF comparing original vs reconstructed images across time steps.

    Args:
        original:      tensor or array [N, C, H, W]
        reconstructed: tensor or array [N, C, H, W]
        file_name:     output path without extension (e.g. "results/sharp_recon")
        plot_name:     title shown on each frame
        channel_names: list of channel name strings, defaults to ["Channel 0", ...]
        num_samples:   number of time steps to include, defaults to all
        cmap:          matplotlib colormap
        figsize:       figure size per frame
        dpi:           resolution per frame — lower = smaller file
        duration:      ms per frame in the GIF
    """
    
    if hasattr(original, 'cpu'):
        original = original.cpu()
    if hasattr(reconstructed, 'cpu'):
        reconstructed = reconstructed.cpu()

    n, c, h, w = original.shape

    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(c)]
    if num_samples is None:
        num_samples = n

    global_min = min(original.min(), reconstructed.min())
    global_max = max(original.max(), reconstructed.max())

    frames = []

    for sample_idx in range(num_samples):
        fig, axes = plt.subplots(2, c, figsize=figsize, dpi=dpi, constrained_layout=True)
        if c == 1:
            axes = axes[:, None]  # ensure 2D indexing for single-channel case
        fig.suptitle(f"{plot_name} — t={sample_idx + 1}", fontsize=12)

        for channel_idx, channel_name in enumerate(channel_names):
            original_channel      = original[sample_idx, channel_idx].numpy()
            reconstructed_channel = reconstructed[sample_idx, channel_idx].numpy()

            ax = axes[0, channel_idx]
            ax.imshow(original_channel, cmap=cmap, aspect="equal",
                      vmin=global_min, vmax=global_max, interpolation="none")
            ax.set_title(f"{channel_name} (GT)", fontsize=10)
            ax.axis("off")

            ax = axes[1, channel_idx]
            ax.imshow(reconstructed_channel, cmap=cmap, aspect="equal",
                      vmin=global_min, vmax=global_max, interpolation="none")
            ax.set_title(f"{channel_name} (Pred)", fontsize=10)
            ax.axis("off")

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        frames.append(Image.fromarray(img[..., :3]))
        plt.close(fig)

    if not frames:
        print("Error: no frames generated.")
        return

    gif_path = file_name + ".gif"
    try:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=duration, loop=0)
        print(f"GIF saved to {gif_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def save_reconstructions_three_gif(original, reconstructed, ground_truth=None,
                              save_path="reconstructions.gif",
                              title="Original vs Reconstructed",
                              channel_names=None, num_samples=None, cmap="jet",
                              figsize=None,
                              show_colorbar=True, clamp=False, fps=2):

    if hasattr(original, 'cpu'):
        original = original.cpu()
    if hasattr(reconstructed, 'cpu'):
        reconstructed = reconstructed.cpu()
    if ground_truth is not None and hasattr(ground_truth, 'cpu'):
        ground_truth = ground_truth.cpu()

    if clamp:
        original      = original.clamp(0, 1) if hasattr(original, 'clamp') else np.clip(original, 0, 1)
        reconstructed = reconstructed.clamp(0, 1) if hasattr(reconstructed, 'clamp') else np.clip(reconstructed, 0, 1)
        if ground_truth is not None:
            ground_truth = ground_truth.clamp(0, 1) if hasattr(ground_truth, 'clamp') else np.clip(ground_truth, 0, 1)

    n, c, h, w = original.shape

    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(c)]
    if num_samples is None:
        num_samples = n

    columns = [("Blurry", original), ("Sharp", reconstructed)]
    if ground_truth is not None:
        columns.append(("Ground Truth", ground_truth))

    global_min = min(t.min() for _, t in columns)
    global_max = max(t.max() for _, t in columns)

    num_cols = len(columns)
    frames = []

    for sample_idx in range(num_samples):

        if figsize is None:
            aspect_ratio = w / h
            fig, axes = plt.subplots(c, num_cols,
                                     figsize=(num_cols * 4 * aspect_ratio, c * 4),
                                     constrained_layout=True)
        else:
            fig, axes = plt.subplots(c, num_cols, figsize=figsize, constrained_layout=True)

        # normalize axes shape
        if c == 1 and num_cols == 1:
            axes = axes[None, None]
        elif c == 1:
            axes = axes[None, :]
        elif num_cols == 1:
            axes = axes[:, None]

        fig.suptitle(f"{title} — Sample {sample_idx + 1}", fontsize=14)

        for channel_idx, channel_name in enumerate(channel_names):
            for col_idx, (col_label, tensor) in enumerate(columns):

                channel_data = tensor[sample_idx, channel_idx].numpy()

                ax = axes[channel_idx, col_idx]
                im = ax.imshow(
                    channel_data,
                    cmap=cmap,
                    aspect="equal",
                    vmin=global_min,
                    vmax=global_max,
                    interpolation="none"  # 🔥 CRITICAL FIX
                )

                ax.set_title(f"{col_label} — {channel_name}", fontsize=10)
                ax.axis("off")

                if show_colorbar:
                    fig.colorbar(im, ax=ax)

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)

        frames.append(Image.fromarray(img[..., :3]))

        plt.close(fig)

    if not frames:
        print("Error: no frames generated.")
        return

    duration_ms = int(1000 / fps)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )

    print(f"GIF saved to {save_path} ({len(frames)} frames @ {fps} fps)")

def plot_sr_losses(log_files, title="Training and Validation Loss", figsize=(10, 6), show_blurry=True, ax=None, label_prefix=""):
    """
    Plot total, sharp, and blurry loss curves on a single plot.
    Args:
        log_files:     list of .json log file paths in sequential order
        title:         figure title
        figsize:       figure size (only used if ax is None)
        show_blurry:   whether to include blurry loss curves (default True)
        ax:            existing matplotlib Axes to plot onto; if None, a new figure is created
        label_prefix:  prefix for legend labels to distinguish multiple models (e.g. "Model A — ")
    """
    training_loss          = []
    validation_loss        = []
    training_sharp_loss    = []
    validation_sharp_loss  = []
    training_blurry_loss   = []
    validation_blurry_loss = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            log_dict = json.load(f)
        training_loss.extend(log_dict['training_loss_per_epoch'])
        validation_loss.extend(log_dict['validation_loss_per_epoch'])
        training_sharp_loss.extend(log_dict['training_sharp_loss_per_epoch'])
        validation_sharp_loss.extend(log_dict['validation_sharp_loss_per_epoch'])
        if show_blurry:
            training_blurry_loss.extend(log_dict['training_blurry_loss_per_epoch'])
            validation_blurry_loss.extend(log_dict['validation_blurry_loss_per_epoch'])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(training_loss) + 1)
    p = label_prefix
    ax.plot(epochs, training_loss,         label=f"{p}Train — total",  color="blue")
    ax.plot(epochs, validation_loss,       label=f"{p}Val — total",    color="blue",   linestyle="--")
    ax.plot(epochs, training_sharp_loss,   label=f"{p}Train — sharp",  color="orange")
    ax.plot(epochs, validation_sharp_loss, label=f"{p}Val — sharp",    color="orange", linestyle="--")
    if show_blurry:
        ax.plot(epochs, training_blurry_loss,   label=f"{p}Train — blurry", color="green")
        ax.plot(epochs, validation_blurry_loss, label=f"{p}Val — blurry",   color="green",  linestyle="--")

    # Vertical lines at checkpoint boundaries
    boundary = 0
    for log_file in log_files[:-1]:
        with open(log_file, 'r') as f:
            boundary += len(json.load(f)['training_loss_per_epoch'])
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, label=f'Resume ({log_file})')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    # plt.show()

def plot_sr_sharp_losses(log_files, title="Training and Validation Sharp Loss", figsize=(10, 6), ax=None, label_prefix=""):
    """
    Plot only the sharp loss curves.
    Args:
        log_files:     list of .json log file paths in sequential order
        title:         figure title
        figsize:       figure size (only used if ax is None)
        ax:            existing matplotlib Axes to plot onto; if None, a new figure is created
        label_prefix:  prefix for legend labels to distinguish multiple models (e.g. "Model A — ")
    """
    training_sharp_loss   = []
    validation_sharp_loss = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            log_dict = json.load(f)
        training_sharp_loss.extend(log_dict['training_sharp_loss_per_epoch'])
        validation_sharp_loss.extend(log_dict['validation_sharp_loss_per_epoch'])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(training_sharp_loss) + 1)
    p = label_prefix
    ax.plot(epochs, training_sharp_loss,   label=f"{p}Train — sharp", color="orange")
    ax.plot(epochs, validation_sharp_loss, label=f"{p}Val — sharp",   color="orange", linestyle="--")

    # Vertical lines at checkpoint boundaries
    boundary = 0
    for log_file in log_files[:-1]:
        with open(log_file, 'r') as f:
            boundary += len(json.load(f)['training_sharp_loss_per_epoch'])
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, label=f'Resume ({log_file})')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
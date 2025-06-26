## eventually want to do ratio plots, etc. here

import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import torch


### NORM RATIO ANALYSIS FUNCTIONS ###

def compute_finite_difference(time_sequence):
    '''
    input: GT or prediction sequence for full reaction (in this notebook is 15ts)
    output: tensor with differences between each ts rather than the actual ts
    '''
    
    n_ts = time_sequence.shape[0]
    
    finite_difference = []
    for i in range(n_ts-1):
        diff = time_sequence[i+1, ...] - time_sequence[i, ...]
        finite_difference.append(diff)
    
    return torch.stack(finite_difference, dim=0)

### NOTE: FOR STEER, CAN JUST DO ABOVE BUT CALC DOT PRODUCT INSTEAD OF DIFFERENCE, INPUT RESULTS FROM ABOVE INTO DOT PROD FUNC

# def calculate_norm(input_data, latent_data, reconstruction_data):
#     '''
#     * Currently uses L2 norm
    
#     Inputs: expected shape [n_ts, channels, H, W]
    
#     Outputs: output is a list with length n_ts of norms for each flattened vector at that time step
#     latent_norm: output is a list with length n_ts of norms for each flattened vector at that time step
#     '''
    
#     n_ts = input_data.shape[0]
    
#     input_data_norm = torch.norm(input_data.view(n_ts, -1), dim=1) 
#     latent_data_norm = torch.norm(latent_data.view(n_ts, -1), dim=1) 
#     reconstruction_data_norm = torch.norm(reconstruction_data.view(n_ts, -1), dim=1) 
    
#     input_norm = input_data_norm.tolist()
#     latent_norm = latent_data_norm.tolist()
#     reconstruction_norm = reconstruction_data_norm.tolist()
    
    
#     return input_norm, latent_norm, reconstruction_norm

def calculate_norm(input_data, latent_data, reconstruction_data):
    '''
    Compute L2 norm over time for each input tensor.

    Inputs:
        input_data: Tensor of shape [time, ...]
        latent_data: Tensor of shape [time, ...]
        reconstruction_data: Tensor of shape [time, ...]
    
    Returns:
        input_norm, latent_norm, reconstruction_norm: Lists of float norms at each time step
    '''
    def norm_over_time(x):
        t = x.shape[0]
        return torch.norm(x.view(t, -1), dim=1).tolist()

    input_norm = norm_over_time(input_data)
    latent_norm = norm_over_time(latent_data)
    reconstruction_norm = norm_over_time(reconstruction_data)

    return input_norm, latent_norm, reconstruction_norm



def calculate_separate_norm(input_data, reconstruction_data):
    '''
    Computes L2 norm per channel over the spatial dimensions [H, W].

    Inputs: expected shape [n_ts, channels, H, W]

    Outputs: Each output is a tensor of shape [n_ts, channels],
             containing norms per channel for each time step.
    '''
    input_norm = torch.norm(input_data, dim=(2, 3))           # shape: [n_ts, channels]
    reconstruction_norm = torch.norm(reconstruction_data, dim=(2, 3))  # shape: [n_ts, channels]

    return input_norm.detach().numpy(), reconstruction_norm.detach().numpy()

# def compute_pairwise_dot_product(time_sequence):
#     '''
#     input: GT or prediction sequence for full reaction (e.g., shape [15, C, H, W])
#     output: tensor with dot products between each consecutive time step
#     '''
    
#     n_ts = time_sequence.shape[0]
    
#     dot_products = []
#     for i in range(n_ts - 1):
#         # Flatten both tensors to compute dot product over entire volume
#         a = time_sequence[i + 1, ...].flatten()
#         b = time_sequence[i, ...].flatten()
#         dot = torch.dot(a, b)
#         dot_products.append(dot)
    
#     return torch.stack(dot_products, dim=0).detach().numpy()

def compute_pairwise_dot_product(time_sequence):
    '''
    input: GT or prediction sequence for full reaction (e.g., shape [15, C, H, W])
    output: tensor with cosine similarity (dot product of unit vectors) between each consecutive time step
    '''
    n_ts = time_sequence.shape[0]
    dot_products = []

    for i in range(n_ts - 1):
        a = time_sequence[i + 1, ...].flatten()
        b = time_sequence[i, ...].flatten()

        # Normalize to unit vectors
        a_norm = a / (a.norm(p=2) + 1e-8)
        b_norm = b / (b.norm(p=2) + 1e-8)

        # Compute dot product (cosine similarity)
        dot = torch.dot(a_norm, b_norm)
        dot_products.append(dot)

    return torch.stack(dot_products, dim=0).detach().numpy()


## ADD MULTIPLE LINES FUNC OR INTEGRATE ABOVE INTO CODE IN NOTEBOOK?

## COMPARISON BETWEEN PIXEL IN LATENT SPACE AND ITS CORR. RECEPTIVE FIELD IN INPUT DIM, COMPARE FOR PIXELS IN DIFF LOCATIONS (ratio between single pixel in latent space and corr. receptive field in input space (compare in diff. regions))

## figure out which samples are outliers observed in all test sample plots


### PLOTTING FUNCTIONS ###

def visualize_reconstructions(X, reconstructions, channel_names, num_ts=5, aspect="auto"):
    """
    Plot original vs reconstructed samples with global color scaling.

    Args:
        X (Tensor): Ground truth data, shape (n_ts, C, H, W).
        reconstructions (Tensor): Reconstructed data, same shape as X.
        channel_names (list of str): Names of each channel (length = C).
        num_samples (int): Number of samples to visualize (default: 5).
    """
    # Compute global min and max for consistent color scaling
    global_min = min(X.min().item(), reconstructions.min().item())
    global_max = max(X.max().item(), reconstructions.max().item())

    for sample_idx in range(num_ts):
        fig, axes = plt.subplots(len(channel_names), 2, figsize=(12, 3 * len(channel_names)))
        fig.suptitle(f"Sample {sample_idx + 1}: Original vs Reconstructed", fontsize=16)

        for channel_idx, channel_name in enumerate(channel_names):
            original_channel = X[sample_idx, channel_idx, :, :].detach().cpu().numpy()
            reconstructed_channel = reconstructions[sample_idx, channel_idx, :, :].detach().cpu().numpy()

            # Plot original
            ax = axes[channel_idx, 0]
            im = ax.imshow(original_channel, cmap="jet", aspect=aspect, vmin=global_min, vmax=global_max)
            ax.set_title(f"Original - {channel_name}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

            # Plot reconstructed
            ax = axes[channel_idx, 1]
            im = ax.imshow(reconstructed_channel, cmap="jet", aspect=aspect, vmin=global_min, vmax=global_max)
            ax.set_title(f"Reconstructed - {channel_name}")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        
def plot_latent_channels_grid(tensor, timestep, n_channels=16, grid_shape=(4, 4), cmap="jet"):
    """
    Plot a grid of latent channel images at a specific timestep.

    Args:
        tensor (Tensor): Input tensor with shape (n_ts, C, H, W).
        timestep (int): Timestep to visualize.
        n_channels (int): Number of channels to display (default: 16).
        grid_shape (tuple): Grid shape (rows, cols) for subplots.
        cmap (str): Colormap for imshow (default: "jet").
    """
    images = tensor[timestep].detach().cpu().numpy()
    rows, cols = grid_shape

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    for idx, ax in enumerate(axes.flat):
        if idx < n_channels:
            im = ax.imshow(images[idx], cmap=cmap)
            ax.set_title(f"Channel {idx}")
            ax.axis("off")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.show()
        
### GIF GENERATION FUNCTIONS ###

def generate_latent_gif(
    tensor,
    file_save_name,
    n_latent_channels=16,
    grid_shape=(4, 4),
    duration=500,
    loop=0,
    cmap="jet",
    max_timesteps=None
):
    """
    Generates a GIF showing the evolution of latent channels over time.

    Args:
        tensor (Tensor): Input tensor of shape (T, C, H, W).
        file_save_name (str): Output GIF filename (without extension).
        n_latent_channels (int): Number of channels to display (default: 16).
        grid_shape (tuple): Tuple of (rows, cols) for subplot layout.
        duration (int): Frame duration in milliseconds.
        loop (int): Number of times the GIF should loop (0 = infinite).
        cmap (str): Colormap to use for visualization.
        max_timesteps (int or None): Optionally limit number of timesteps to include.
    """
    # Convert tensor to numpy
    all_data = tensor.cpu().detach().numpy()  # Shape: (T, C, H, W)
    n_ts = all_data.shape[0]
    if max_timesteps is not None:
        n_ts = min(n_ts, max_timesteps)

    # Step 1: Compute per-channel vmin and vmax
    channel_vmin = all_data.min(axis=(0, 2, 3))
    channel_vmax = all_data.max(axis=(0, 2, 3))

    print("Per-channel vmin and vmax computed.")

    frames = []

    # Step 2: Create frames
    for ts in range(n_ts):
        images = all_data[ts]

        fig, axes = plt.subplots(*grid_shape, figsize=(4 * grid_shape[1], 3 * grid_shape[0]))

        for idx, ax in enumerate(axes.flat):
            if idx < n_latent_channels:
                im = ax.imshow(
                    images[idx],
                    cmap=cmap,
                    vmin=channel_vmin[idx],
                    vmax=channel_vmax[idx]
                )
                ax.set_title(f"Channel {idx}")
                ax.axis("off")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
            else:
                ax.axis("off")

        plt.tight_layout()
        fig.canvas.draw()

        # Convert canvas to numpy array (RGB)
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[..., :3]  # Drop alpha channel
        frames.append(frame)

        plt.close(fig)

    # Step 3: Save the GIF
    imageio.mimsave(f'{file_save_name}.gif', frames, duration=duration, loop=loop)
    print(f"GIF saved as '{file_save_name}.gif'")


def generate_prediction_comparison_gif(
    X,
    prediction,
    channel_names,
    file_name,
    plot_title,
    duration=500,
    dpi=150
):
    """
    Creates a GIF comparing original and reconstructed channels over time.

    Args:
        X (Tensor or ndarray): Ground truth data of shape (T, C, H, W).
        prediction (Tensor or ndarray): Predicted data of shape (T, C, H, W).
        channel_names (list of str): Names of each channel.
        file_name (str): Name for the output GIF file (without extension).
        plot_title (str): Title to display on each frame.
        duration (int): Duration of each frame in milliseconds (default: 500).
        dpi (int): Resolution of each frame (default: 150).
    """
    # Convert to numpy if tensor
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()

    num_samples = X.shape[0]
    num_channels = X.shape[1]

    # Compute global vmin and vmax
    global_min = min(X.min(), prediction.min())
    global_max = max(X.max(), prediction.max())

    frames = []

    for sample_idx in range(num_samples):
        fig, axes = plt.subplots(2, num_channels, figsize=(6, 4), dpi=dpi, constrained_layout=True)
        fig.suptitle(plot_title, fontsize=12)

        for channel_idx, channel_name in enumerate(channel_names):
            original_channel = X[sample_idx, channel_idx, :, :]
            reconstructed_channel = prediction[sample_idx, channel_idx, :, :]

            # Top row: Ground truth
            ax = axes[0, channel_idx]
            ax.imshow(original_channel, cmap="jet", aspect="equal",
                      vmin=global_min, vmax=global_max, interpolation="none")
            ax.set_title(f"{channel_name} (GT)", fontsize=10)
            ax.axis("off")

            # Bottom row: Prediction
            ax = axes[1, channel_idx]
            ax.imshow(reconstructed_channel, cmap="jet", aspect="equal",
                      vmin=global_min, vmax=global_max, interpolation="none")
            ax.set_title(f"{channel_name} (Pred)", fontsize=10)
            ax.axis("off")

        # Convert the figure to a PIL Image and add to frames
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(Image.fromarray(img))
        plt.close(fig)

    if frames:
        gif_path = file_name + ".gif"
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF saved to {gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print("Error: No frames generated!")


import json
import os
import torch

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
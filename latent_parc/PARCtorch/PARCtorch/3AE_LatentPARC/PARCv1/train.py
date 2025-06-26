import torch
import numpy as np
from tqdm import tqdm
import json

def save_model_and_logs(model, save_path, weights_name, log_dict, epochs):
    """Save model weights and training logs."""
    if save_path:
        torch.save(model.state_dict(), f"{save_path}/{weights_name}_{epochs}.pth")
        with open(f"{save_path}/{weights_name}_{epochs}.json", 'w') as f:
            json.dump(log_dict, f)
        print(f"Model and logs saved to {save_path}")
            
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

def train_parc(model, optimizer, loss_function, train_loader, val_loader, device, 
                      epochs=10, image_size=(128, 256), n_channels=3, scheduler=None, 
                      noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5,
                      reduce_on=500, loss_weights=[1.0,1.0,1.0],
                      mode="single_ts_train", save_path=None, weights_name=None):
    """
    Train LatentPARC model.
    
    model: lp_model
    optimizer: optimizer
    loss_function: loss function
    train_loader: training data
    val_loader: validation data
    device: GPU
    epochs: number of epochs to train for
    imagesize: tuple, (height, width)
    n_channels: T,P,ms etc. however many of these (=5 to include velocity)
    scheduler: loss scheduler
    noise_fn: function for adding noise to images
    initial_max_noise: max magnitude of noise to add
    n_reduce_factor: fraction to reduce noise by at each reduce on epoch
    ms_reduce factor: e.g. if you want to decrease magnitude of ms by .5, set equal to .5
    reduce_on: epoch interval to decrease noise
    loss_weights: weights to apply to purple, orange, green loss -> key in slides, need to rename in code
    mode: options "single_ts_train" or "rollout_train", currently rollout train is only for frozen or preload AE
    save_path: where to save weights and loss history
    weights name: name of the weights
    """
    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': []}
    
    model.to(device)

    max_noise = initial_max_noise  # Initialize noise level

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reduce noise periodically
        if (epoch + 1) % reduce_on == 0:
            max_noise *= n_reduce_factor

        # --- Training ---
        model.train()
        train_losses = []
        
        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            ic, _, n_ts, target = images # ic is first frame of subsequence not always actual IC, n_ts is total ts - 1           
            
            # RESHAPE
            ic = ic[:, :n_channels, ...].to(device)
            n_ts = n_ts.shape[0]
            target = target[:, :, :n_channels, ...].to(device)

            ic = noise_fn(ic, max_val=max_noise) if noise_fn else ic  # optional add noise step

            if mode == "single_ts_train":
                assert n_ts == 1, "In 'single_ts_train' mode, n_ts must be 1"
                
                target = target.squeeze(0) # no longer need n_ts channel
                pred = model(ic)
                loss = loss_function(pred, target)
            
            elif mode == "rollout_train":  
                pred = model(ic, n_ts=n_ts+1, mode='pred') # n_ts + 1 b/c also outputs IC z and recon. 0
                loss = loss_function(pred[1:, ...], target)
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        log_dict['training_loss_per_epoch'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_ic, _, n_ts, val_target = val_images # ic is first frame of subseq. not always actual IC, ts is number of ts - 1   
                
                # RESHAPE
                val_ic = val_ic[:, :n_channels, ...].to(device)
                n_ts = n_ts.shape[0]
                val_target = val_target[:, :, :n_channels, ...].to(device)
                    
                if mode == "single_ts_train":
                    assert n_ts == 1, "In 'single_ts_train' mode, n_ts must be 1"

                    val_target = val_target.squeeze(0) # no longer need n_ts channel
                
                    val_pred = model(val_ic, n_ts=1) 
                    val_loss = loss_function(val_pred, val_target)

                elif mode == "rollout_train":
                    val_pred = model(val_ic, n_ts=n_ts+1, mode='pred') # n_ts + 1 b/c also outputs IC z and recon.
                    val_loss = loss_function(val_pred[1:, ...], val_target)

                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

    # Save model and logs
    save_model_and_logs(model, save_path, weights_name, log_dict, epochs)

    return log_dict

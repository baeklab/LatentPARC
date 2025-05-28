import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class DifferentiatorEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        in_channels = 3
        layer_depths = [8, 32, 64, 32]
        
        # block 1
        self.b1c1 = nn.Conv2d(in_channels, layer_depths[0], kernel_size=3, padding=1)  
        self.b1c2 = nn.Conv2d(layer_depths[0], layer_depths[0], kernel_size=3, padding=1)
        self.b1c3 = nn.Conv2d(layer_depths[0], layer_depths[0], kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2
        self.b2c1 = nn.Conv2d(layer_depths[0], layer_depths[1], kernel_size=3, padding=1)
        self.b2c2 = nn.Conv2d(layer_depths[1], layer_depths[1], kernel_size=3, padding=1)
        self.b2c3 = nn.Conv2d(layer_depths[1], layer_depths[1], kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 3
        self.b3c1 = nn.Conv2d(layer_depths[1], layer_depths[1], kernel_size=7, padding=3)
        self.b3c2 = nn.Conv2d(layer_depths[1], layer_depths[2], kernel_size=1)
        self.b3c3 = nn.Conv2d(layer_depths[2], layer_depths[3], kernel_size=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # output projection (optional)
        self.out = nn.Conv2d(layer_depths[3], latent_dim, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Block 1 + pooling
        b1c1 = self.leaky_relu(self.b1c1(x))
        b1c2 = self.leaky_relu(self.b1c2(b1c1))
        b1c3 = self.b1c3(b1c2)
        b1_out = self.relu(b1c1 + b1c3)
        b1_out = self.pool1(b1_out)

        # Block 2 + pooling
        b2c1 = self.leaky_relu(self.b2c1(b1_out))
        b2c2 = self.leaky_relu(self.b2c2(b2c1))
        b2c3 = self.b2c3(b2c2)
        b2_out = self.relu(b2c1 + b2c3)
        b2_out = self.pool2(b2_out)

        # Block 3 + pooling
        b3c1 = self.leaky_relu(self.b3c1(b2_out))
        b3c2 = self.leaky_relu(self.b3c2(b3c1))
        b3c3 = self.dropout(self.b3c3(b3c2))
        b3_out = self.leaky_relu(b3c3)
        b3_out = self.pool3(b3_out)

        # Output projection (no activation to keep flexibility)
        out = self.out(b3_out)
        
        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        layer_depths = [32, 64, 128, 64]  # reverse of encoder

        self.up1 = nn.Sequential(
            nn.Conv2d(latent_dim, layer_depths[0], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(layer_depths[0], layer_depths[1], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(layer_depths[1], layer_depths[2], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Optional refinement after last upsample
        self.final = nn.Sequential(
            nn.Conv2d(layer_depths[2], layer_depths[3], kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(layer_depths[3], latent_dim, kernel_size=3, padding=1),  # match input channels
        )

    def forward(self, z):
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        out = self.final(x)
        return out

class model(nn.Module):
    def __init__(self, encoder, decoder, integrator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        if mode == 'train':
            if n_ts == 0: # case for autoencoder reconstruction only, no dynamics
                z = self.encoder(x)
                decoded = self.decoder(z)
            elif n_ts == 1: 
                z_init = self.encoder(x)
                z, _ = self.integrator(self.encoder, 0.0, z_init, 0.1)
                decoded = self.decoder(z)
        
        elif mode == 'pred': 
            z_list, decoded_list = [], []
            z_i = self.encoder(x)
            z_list.append(z_i)
            decoded_list.append(self.decoder(z_i))
          
            for i in range(n_ts-1):
                z_i, _ = self.integrator(self.encoder, 0.0, z_list[i], 0.1)
                z_list.append(z_i)
                decoded_list.append(self.decoder(z_i))
            z, decoded = torch.stack(z_list, dim=0), torch.stack(decoded_list, dim=0)
        return z, decoded
    
    
# ------------
# TRAINING FUNCTION
# ------------
def train_latentparc(model, optimizer, loss_function, train_loader, val_loader, device, 
                      epochs=10, image_size=(128, 256), n_channels=3, scheduler=None, 
                      noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5,
                      ms_reduce_factor=0, reduce_on=500, loss_weights=[1.0,1.0,1.0],
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
    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': [],
               'purple_loss_train': [], 
               'green_loss_train': [], 
               'orange_loss_train': []}
    
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
        purple_losses = []
        green_losses = []
        orange_losses = []
        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # UNPACK
            # ic.shape -> [batch, n_channel, H, W]
            # target.shape -> [timestep, batch, n_channel, H, W] note: timestep goes from 1->ts with 0 being ic
            ic, _, n_ts, target = images # ic is first frame of subsequence not always actual IC, n_ts is total ts - 1           
            
            # RESHAPE
            ic = ic[:, :n_channels, ...].to(device)
            n_ts = n_ts.shape[0]
            target = target[:, :, :n_channels, ...].to(device)
        
            # rollout_GT = torch.concat((ic.unsqueeze(0), target)) # concat ic and the rest of the timesteps, may switch all ic and target to this for simplicity
                
            if ms_reduce_factor != 0: # Optional: reduce microstructure magnitude
                ic[:, 2, ...] *= ms_reduce_factor
                target[:, :, 2, ...] *= ms_reduce_factor

            ic = noise_fn(ic, max_val=max_noise) if noise_fn else ic  # optional add noise step

            if mode == "single_ts_train":
                assert n_ts == 1, "In 'single_ts_train' mode, n_ts must be 1"
                
                target = target.squeeze(0) # no longer need n_ts channel

                _, x_bar = model(ic, n_ts=0)
                z_hat, x_hat = model(ic, n_ts=1) 
                z_hat_next, _ = model(target, n_ts=0) 

                L_1 = loss_function(x_bar, ic)
                L_2 = loss_function(z_hat, z_hat_next)
                L_3 = loss_function(x_hat, target)
            
            elif mode == "rollout_train":
                # collect GT latent encodings for all target ts (will be ts 1->n_ts, ts 0 latent no dynamics)
                z_hat_next = []
                for i in range(n_ts): # only have GT for n_ts latent next fields (there are n_ts + 1 steps with ic)
                    z_hat_next_i, _ = model(target[i, ...], n_ts=0)
                    z_hat_next.append(z_hat_next_i)
                z_hat_next = torch.stack(z_hat_next, dim=0)
                
                z_hat, x_hat = model(ic, n_ts=n_ts+1, mode='pred') # n_ts + 1 b/c also outputs IC z and recon. 0
                
                L_1 = torch.tensor(0.0, device=device) # don't need recon. loss, can do with x_hat[0, ...] though
                L_2 = loss_function(z_hat[1:, ...], z_hat_next)
                L_3 = loss_function(x_hat[1:, ...], target)
            
            loss = (L_1*loss_weights[0] + 
                    L_2*loss_weights[1] + 
                    L_3*loss_weights[2])
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            purple_losses.append(L_1.item())
            green_losses.append(L_2.item())
            orange_losses.append(L_3.item())

        avg_train_loss = np.mean(train_losses)
        avg_purple_loss = np.mean(purple_losses) # reconstruction loss
        avg_green_loss = np.mean(green_losses) # latent dynamics loss
        avg_orange_loss = np.mean(orange_losses) # reconstructed + dynamics loss
        
        log_dict['training_loss_per_epoch'].append(avg_train_loss)
        log_dict['purple_loss_train'].append(avg_purple_loss)
        log_dict['green_loss_train'].append(avg_green_loss)
        log_dict['orange_loss_train'].append(avg_orange_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                # UNPACK
                # ic.shape -> [batch, n_channel, H, W]
                # target.shape -> [timestep, batch, n_channel, H, W] note: timestep goes from 1->ts with 0 being ic
                val_ic, _, n_ts, val_target = val_images # ic is first frame of subsequence not always actual IC, ts is number of ts - 1           
            
                # RESHAPE
                val_ic = val_ic[:, :n_channels, ...].to(device)
                n_ts = n_ts.shape[0]
                val_target = val_target[:, :, :n_channels, ...].to(device)
                
                if ms_reduce_factor != 0: # Optional: reduce microstructure magnitude
                    val_ic[:, 2, ...] *= ms_reduce_factor
                    val_target[:, :, 2, ...] *= ms_reduce_factor
                    
                if mode == "single_ts_train":
                    assert n_ts == 1, "In 'single_ts_train' mode, n_ts must be 1"

                    val_target = val_target.squeeze(0) # no longer need n_ts channel

                    _, x_bar = model(val_ic, n_ts=0)
                    z_hat, x_hat = model(val_ic, n_ts=1) 
                    z_hat_next, _ = model(val_target, n_ts=0) 

                    val_L_1 = loss_function(x_bar, val_ic)
                    val_L_2 = loss_function(z_hat, z_hat_next)
                    val_L_3 = loss_function(x_hat, val_target)

                elif mode == "rollout_train":
                    # collect GT latent encodings for all target ts (will be ts 1->n_ts, ts 0 latent no dynamics)
                    z_hat_next = []
                    for i in range(n_ts): # only have GT for n_ts latent fields (there are n_ts + 1 steps with ic)
                        z_hat_next_i, _ = model(val_target[i, ...], n_ts=0)
                        z_hat_next.append(z_hat_next_i)
                    z_hat_next = torch.stack(z_hat_next, dim=0)

                    z_hat, x_hat = model(val_ic, n_ts=n_ts+1, mode='pred') # n_ts + 1 b/c also outputs IC z and recon.

                    val_L_1 = torch.tensor(0.0, device=device) # don't need recon. loss, can do with x_hat[0, ...] though
                    val_L_2 = loss_function(z_hat[1:, ...], z_hat_next)
                    val_L_3 = loss_function(x_hat[1:, ...], val_target)


                val_loss = (val_L_1*loss_weights[0] + 
                        val_L_2*loss_weights[1] + 
                        val_L_3*loss_weights[2])

                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

    # Save model and logs
    save_model_and_logs(model, save_path, weights_name, log_dict, epochs)

    return log_dict




# ------------
# LATENT PARC CLASS (WRAPPER)
# ------------
class LatentPARC:
    def __init__(self, lp_model, optimizer, save_path=None, weights_name=None):
        self.network = lp_model
        self.optimizer = optimizer
        self.save_path = save_path
        self.weights_name = weights_name

    def train(self, loss_function, epochs, image_size, n_channels, device, train_loader, val_loader, scheduler=None, noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5, ms_reduce_factor=0, reduce_on=1000, loss_weights=[1.0, 1.0, 1.0], mode="single_ts_train"):
        """Wrapper for training."""
        return train_latentparc(self.network, self.optimizer, loss_function, train_loader, val_loader,
                                device, epochs, image_size, n_channels, scheduler, noise_fn,
                                initial_max_noise, n_reduce_factor, ms_reduce_factor, reduce_on,
                                loss_weights, mode, self.save_path, self.weights_name)

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        return self.network.encoder(x)

    def decode(self, x):
        return self.network.decoder(x)
    
    
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



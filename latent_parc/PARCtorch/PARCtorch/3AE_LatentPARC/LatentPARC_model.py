import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
import torch.nn.functional as F
from PARCv1.differentiator import *
from autoencoder.autoencoder import *


# Helper Functions
class LpLoss(torch.nn.Module):
    def __init__(self, p=10):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, input, target):
        # Compute element-wise absolute difference
        diff = torch.abs(input - target)
        # Raise the differences to the power of p, sum them, and raise to the power of 1/p
        return (torch.sum(diff ** self.p) ** (1 / self.p))

def init_weights(module):
    """Initialize weights using Xavier uniform initialization."""
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


def save_model_and_logs(model, save_path, weights_name, log_dict, epochs):
    """Save model weights and training logs."""
    if save_path:
        torch.save(model.state_dict(), f"{save_path}/{weights_name}_{epochs}.pth")
        with open(f"{save_path}/{weights_name}_{epochs}.json", 'w') as f:
            json.dump(log_dict, f)
        print(f"Model and logs saved to {save_path}")


# PERCEPTUAL LOSS FUNCTIONS
# Load Pretrained Feature Extractor (VGG19)
class FeatureExtractor(nn.Module):
    def __init__(self, layers=[2, 7, 12]):  # Example layers: relu1_2, relu2_2, relu3_2
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features  # Updated weights usage
        self.selected_layers = layers
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:max(layers) + 1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features

# Perceptual Loss Function
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, loss_fn=nn.L1Loss()):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn  # Can use L1 or MSE loss

    def forward(self, input, target):
        with torch.no_grad():  # Save memory by disabling gradient computation
            input_features = self.feature_extractor(input)
            target_features = self.feature_extractor(target)
        
        loss = sum(self.loss_fn(inp, tgt) for inp, tgt in zip(input_features, target_features))
        return loss


# ------------
# LATENT PARC MODEL
# ------------
# Original model where all 3 fields go into 1 autoencoder
class lp_model(nn.Module):
    def __init__(self, encoder, decoder, differentiator, integrator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.differentiator = differentiator
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        if mode == 'train':
            if n_ts == 0:
                z = self.encoder(x)
                decoded = self.decoder(z)
            elif n_ts == 1:
                z_init = self.encoder(x)
                z, _ = self.integrator(self.differentiator, 0.0, z_init, 0.1)
                decoded = self.decoder(z)
        elif mode == 'pred':
            z_list, decoded_list = [], []
            z_i = self.encoder(x)
            z_list.append(z_i)
            decoded_list.append(self.decoder(z_i))
            for _ in range(n_ts - 1):
                z_i, _ = self.integrator(self.differentiator, 0.0, z_i, 0.1)
                z_list.append(z_i)
                decoded_list.append(self.decoder(z_i))
            z, decoded = torch.stack(z_list, dim=0), torch.stack(decoded_list, dim=0)
        return z, decoded

# model where ms goes into a separate AE, T and P are still encoded together, concat in latent space
class lp_model_2plus1_1decoder(nn.Module):
    def __init__(self, encoder_TP, encoder_M, decoder, differentiator, integrator):
        super().__init__()
        self.encoderTP = encoder_TP
        self.encoderM = encoder_M
        self.decoder = decoder
        self.differentiator = differentiator
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        if mode == 'train':
            if n_ts == 0:
                z_tp = self.encoderTP(x[:, :2, :, :]) # only T,P channels
                z_m = self.encoderM(x[:, 2:3, :, :]) # only M channel
                z = torch.cat((z_tp, z_m), dim=1) # concat in latent space
                decoded = self.decoder(z) # decode with 1 decoder for all 3 channels
            elif n_ts == 1:
                z_tp_init = self.encoderTP(x[:, :2, :, :]) # encode T, P
                z_m_init = self.encoderM(x[:, 2:3, :, :]) # encode M separately
                z_init = torch.cat((z_tp_init, z_m_init), dim=1) # concat in latent space
                z, _ = self.integrator(self.differentiator, 0.0, z_init, 0.1)
                decoded = self.decoder(z) # decode all together
        elif mode == 'pred': #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOT EDITED YET, WAITING TO SEE MODE TRAIN WORK
            z_list, decoded_list = [], []
            z_i = self.encoder(x)
            z_list.append(z_i)
            decoded_list.append(self.decoder(z_i))
            for _ in range(n_ts - 1):
                z_i, _ = self.integrator(self.differentiator, 0.0, z_i, 0.1)
                z_list.append(z_i)
                decoded_list.append(self.decoder(z_i))
            z, decoded = torch.stack(z_list, dim=0), torch.stack(decoded_list, dim=0)
        return z, decoded

# model where T, P, ms all go into separate encoders, concat in latent space, decode together
class lp_model_3encoder_1decoder(nn.Module):
    def __init__(self, encoder_T, encoder_P, encoder_M, decoder, differentiator, integrator):
        super().__init__()
        self.encoderT = encoder_T
        self.encoderP = encoder_P
        self.encoderM = encoder_M
        self.decoder = decoder
        self.differentiator = differentiator
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        if mode == 'train':
            if n_ts == 0:
                z_t = self.encoderT(x[:, 0:1, :, :]) # only T channel
                z_p = self.encoderP(x[:, 1:2, :, :]) # only P channel
                z_m = self.encoderM(x[:, 2:3, :, :]) # only M channel
                z = torch.cat((z_t, z_p, z_m), dim=1) # concat in latent space
                decoded = self.decoder(z) # decode with 1 decoder for all 3 channels
            elif n_ts == 1:
                z_t_init = self.encoderT(x[:, 0:1, :, :]) # encode T
                z_p_init = self.encoderP(x[:, 1:2, :, :]) # encode P
                z_m_init = self.encoderM(x[:, 2:3, :, :]) # encode M
                z_init = torch.cat((z_t_init, z_p_init, z_m_init), dim=1) # concat in latent space
                z, _ = self.integrator(self.differentiator, 0.0, z_init, 0.1)
                decoded = self.decoder(z) # decode all together
        elif mode == 'pred': #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOT EDITED YET, WAITING TO SEE MODE TRAIN WORK
            z_list, decoded_list = [], []
            z_i = self.encoder(x)
            z_list.append(z_i)
            decoded_list.append(self.decoder(z_i))
            for _ in range(n_ts - 1):
                z_i, _ = self.integrator(self.differentiator, 0.0, z_i, 0.1)
                z_list.append(z_i)
                decoded_list.append(self.decoder(z_i))
            z, decoded = torch.stack(z_list, dim=0), torch.stack(decoded_list, dim=0)
        return z, decoded
    
# model where T, P, ms all go into separate encoders, concat in latent space, decode separately
class lp_model_2plus1_2decoder(nn.Module):
    def __init__(self, encoder_TP, encoder_M, decoder_TP, decoder_M, differentiator, integrator):
        super().__init__()
        self.encoderTP = encoder_TP
        self.encoderM = encoder_M
        self.decoderTP = decoder_TP
        self.decoderM = decoder_M
        self.differentiator = differentiator
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        if mode == 'train':
            if n_ts == 0:
                z_tp = self.encoderTP(x[:, 0:2, :, :]) # T, P channels
                z_m = self.encoderM(x[:, 2:3, :, :]) # only M channel
                z = torch.cat((z_tp, z_m), dim=1) # concat in latent space
                
                decoded_tp = self.decoderTP(z[:, 0:8, :, :]) # decode TP
                decoded_m = self.decoderM(z[:, 8:, :, :]) # decode M
                decoded = torch.cat((decoded_tp, decoded_m), dim=1) # concat for output
                
            elif n_ts == 1:
                z_tp_init = self.encoderTP(x[:, 0:2, :, :]) # encode T
                z_m_init = self.encoderM(x[:, 2:3, :, :]) # encode M
                z_init = torch.cat((z_tp_init, z_m_init), dim=1) # concat in latent space
                z, _ = self.integrator(self.differentiator, 0.0, z_init, 0.1)
                
                decoded_tp = self.decoderTP(z[:, 0:8, :, :]) # decode T
                decoded_m = self.decoderM(z[:, 8:, :, :]) # decode M
                decoded = torch.cat((decoded_tp, decoded_m), dim=1) # concat for output
            
        elif mode == 'pred': #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOT EDITED YET, WAITING TO SEE MODE TRAIN WORK
            z_list, decoded_list = [], []
            z_i = self.encoder(x)
            z_list.append(z_i)
            decoded_list.append(self.decoder(z_i))
            for _ in range(n_ts - 1):
                z_i, _ = self.integrator(self.differentiator, 0.0, z_i, 0.1)
                z_list.append(z_i)
                decoded_list.append(self.decoder(z_i))
            z, decoded = torch.stack(z_list, dim=0), torch.stack(decoded_list, dim=0)
        return z, decoded
    
# model where T, P, ms all go into separate encoders, concat in latent space, decode separately
class lp_model_3encoder_3decoder(nn.Module):
    def __init__(self, encoder_T, encoder_P, encoder_M, decoder_T, decoder_P, decoder_M, differentiator, integrator):
        super().__init__()
        self.encoderT = encoder_T
        self.encoderP = encoder_P
        self.encoderM = encoder_M
        self.decoderT = decoder_T
        self.decoderP = decoder_P
        self.decoderM = decoder_M
        self.differentiator = differentiator
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        if mode == 'train':
            if n_ts == 0:
                z_t = self.encoderT(x[:, 0:1, :, :]) # only T channel
                z_p = self.encoderP(x[:, 1:2, :, :]) # only P channel
                z_m = self.encoderM(x[:, 2:3, :, :]) # only M channel
                z = torch.cat((z_t, z_p, z_m), dim=1) # concat in latent space
                
                decoded_t = self.decoderT(z[:, 0:4, :, :]) # decode T
                decoded_p = self.decoderP(z[:, 4:8, :, :]) # decode P
                decoded_m = self.decoderM(z[:, 8:, :, :]) # decode M
                decoded = torch.cat((decoded_t, decoded_p, decoded_m), dim=1) # concat for output
                
            elif n_ts == 1:
                z_t_init = self.encoderT(x[:, 0:1, :, :]) # encode T
                z_p_init = self.encoderP(x[:, 1:2, :, :]) # encode P
                z_m_init = self.encoderM(x[:, 2:3, :, :]) # encode M
                z_init = torch.cat((z_t_init, z_p_init, z_m_init), dim=1) # concat in latent space
                z, _ = self.integrator(self.differentiator, 0.0, z_init, 0.1)
                
                decoded_t = self.decoderT(z[:, 0:4, :, :]) # decode T
                decoded_p = self.decoderP(z[:, 4:8, :, :]) # decode P
                decoded_m = self.decoderM(z[:, 8:, :, :]) # decode M
                decoded = torch.cat((decoded_t, decoded_p, decoded_m), dim=1) # concat for output
            
        elif mode == 'pred': #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NOT EDITED YET, WAITING TO SEE MODE TRAIN WORK
            z_list, decoded_list = [], []
            z_i = self.encoder(x)
            z_list.append(z_i)
            decoded_list.append(self.decoder(z_i))
            for _ in range(n_ts - 1):
                z_i, _ = self.integrator(self.differentiator, 0.0, z_i, 0.1)
                z_list.append(z_i)
                decoded_list.append(self.decoder(z_i))
            z, decoded = torch.stack(z_list, dim=0), torch.stack(decoded_list, dim=0)
        return z, decoded
    
# ------------
# TRAINING FUNCTION
# ------------
def train_latentparc(model, optimizer, loss_function, train_loader, val_loader, device, 
                      epochs=10, image_size=(64, 64), n_channels=3, scheduler=None, 
                      noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5,
                      ms_reduce_factor=0, reduce_on=1000, loss_weights=[1.0,1.0,1.0],
                      save_path=None, weights_name=None):
    """
    Train LatentPARC model.
    
    model: lp_model
    optimizer: 
    loss_function:
    train_loader: training data
    val_loader: validation data
    device: GPU
    epochs: num epochs to train for
    imagesize:
    n_channels: T,P,mu etc. however many of these
    scheduler: loss scheduler
    noise_fn: function for adding noise to images
    initial_max_noise: max magnitude of noise to add
    n_reduce_factor: fraction to reduce noise by at each reduce on epoch
    ms_reduce factor: e.g. if you want to decrease magnitude of ms by .5, set equal to .5
    reduce_on: epoch interval to decrease noise
    save_path: where to save weights and loss history
    weights name:
    """
    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': [],
               'purple_loss_train': [], 
               'green_loss_train': [], 
               'orange_loss_train': []}
    model.apply(init_weights)  # Initialize weights
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
            ic, _, _, target = images
            ic, target = ic[:, :n_channels, ...].to(device), target.squeeze(0)[:, :n_channels, ...].to(device)
            
            # HALVE MICROSTRUCTURE
            if ms_reduce_factor != 0:
                ic[:, 2, ...] *= ms_reduce_factor
                target[:, 2, ...] *= ms_reduce_factor

            noisy_ic = noise_fn(ic, max_val=max_noise) if noise_fn else ic
            noisy_target = noise_fn(target, max_val=max_noise) if noise_fn else target

            _, x_bar = model(noisy_ic, n_ts=0)
            z_hat, x_hat = model(noisy_ic, n_ts=1)
            z_hat_next, _ = model(noisy_target, n_ts=0)

            L_1 = loss_function(x_bar, ic)
            L_2 = loss_function(z_hat, z_hat_next)
            L_3 = loss_function(x_hat, target)
            
            # print(f"Purple: {L_1}, Green: {L_2}, Orange: {L_3}")
            
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
        avg_purple_loss = np.mean(purple_losses)
        avg_green_loss = np.mean(green_losses)
        avg_orange_loss = np.mean(orange_losses)
        
        log_dict['training_loss_per_epoch'].append(avg_train_loss)
        log_dict['purple_loss_train'].append(avg_purple_loss)
        log_dict['green_loss_train'].append(avg_green_loss)
        log_dict['orange_loss_train'].append(avg_orange_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_ic, _, _, val_target = val_images
                val_ic, val_target = val_ic[:, :n_channels, ...].to(device), val_target.squeeze(0)[:, :n_channels, ...].to(device)

                # Halve MS
                if ms_reduce_factor != 0:
                    val_ic[:, 2, ...] *= ms_reduce_factor 
                    val_target[:, 2, ...] *= ms_reduce_factor

                
                _, x_bar = model(val_ic, n_ts=0)
                z_hat, x_hat = model(val_ic, n_ts=1)
                z_hat_next, _ = model(val_target, n_ts=0)
                
                val_L_1 = loss_function(x_bar, val_ic)
                val_L_2 = loss_function(z_hat, z_hat_next)
                val_L_3 = loss_function(x_hat, val_target)

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

    def train(self, loss_function, epochs, image_size, n_channels, device, train_loader, val_loader, scheduler=None, noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5, ms_reduce_factor=0, reduce_on=1000, loss_weights=[1.0, 1.0, 1.0]):
        """Wrapper for training."""
        return train_latentparc(self.network, self.optimizer, loss_function, train_loader, val_loader,
                                device, epochs, image_size, n_channels, scheduler, noise_fn,
                                initial_max_noise, n_reduce_factor, ms_reduce_factor, reduce_on,
                                loss_weights,
                                self.save_path, self.weights_name)

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        return self.network.encoder(x)

    def decode(self, x):
        return self.network.decoder(x)


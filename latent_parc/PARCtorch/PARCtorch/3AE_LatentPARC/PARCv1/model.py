# NOTE: This is my simplified version of PARCv1 used in LatentPARC. For purpose of comparison to LatentPARC w/ AE, not to replicate PARCv1 from the paper.

import torch
import torch.nn as nn
from train import train_parc
     
### Just PARCv1 dynamics module        
class parc_model(nn.Module):
    def __init__(self, differentiator, integrator):
        super().__init__()
        self.differentiator = differentiator
        self.integrator = integrator

    def forward(self, x, n_ts=1, mode='train'):
        if mode == 'train':
            pred, _ = self.integrator(self.differentiator, 0.0, x, 0.1)
        
        elif mode == 'pred': 
            pred_list = []
            pred_list.append(x)
            for i in range(n_ts-1):
                pred, _ = self.integrator(self.differentiator, 0.0, pred_list[i], 0.1)
                pred_list.append(pred)
            pred = torch.stack(pred_list, dim=0)
        return pred
    

# ------------
# PARC CLASS (WRAPPER)
# ------------
class PARC:
    def __init__(self, parc_model, optimizer, save_path=None, weights_name=None):
        self.network = parc_model
        self.optimizer = optimizer
        self.save_path = save_path
        self.weights_name = weights_name

    def train(self, loss_function, epochs, image_size, n_channels, device, train_loader, val_loader, scheduler=None, noise_fn=None, initial_max_noise=0.16, n_reduce_factor=0.5, reduce_on=1000, loss_weights=[1.0, 1.0, 1.0], mode="single_ts_train"):
        """Wrapper for training."""
        return train_parc(self.network, self.optimizer, loss_function, train_loader, val_loader,
                                device, epochs, image_size, n_channels, scheduler, noise_fn,
                                initial_max_noise, n_reduce_factor, reduce_on,
                                loss_weights, mode, self.save_path, self.weights_name)
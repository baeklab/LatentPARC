import torch
import torch.nn as nn
import torch.nn.functional as F    

class Differentiator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        in_channels = latent_dim
        layer_depths = [64, 128, 64, 32]
        
        # block 1
        self.b1c1 = nn.Conv2d(in_channels, layer_depths[0], kernel_size=3, padding=1)  
        self.b1c2 = nn.Conv2d(layer_depths[0], layer_depths[0], kernel_size=3, padding=1)
        self.b1c3 = nn.Conv2d(layer_depths[0], layer_depths[0], kernel_size=3, padding=1)
        
        # block 2
        self.b2c1 = nn.Conv2d(layer_depths[0], layer_depths[1], kernel_size=3, padding=1)
        self.b2c2 = nn.Conv2d(layer_depths[1], layer_depths[1], kernel_size=3, padding=1)
        self.b2c3 = nn.Conv2d(layer_depths[1], layer_depths[1], kernel_size=3, padding=1)
        
        #block 3
        self.b3c1 = nn.Conv2d(layer_depths[1], layer_depths[1], kernel_size=7, padding=3)
        self.b3c2 = nn.Conv2d(layer_depths[1], layer_depths[2], kernel_size=1, padding=0)
        self.b3c3 = nn.Conv2d(layer_depths[2], layer_depths[3], kernel_size=1, padding=0)
        
        self.out = nn.Conv2d(layer_depths[3], latent_dim, kernel_size=3, padding=1)
        
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # self.tanh = nn.Tanh()

    def forward(self, t, x):
        b1c1 = self.leaky_relu(self.b1c1(x))
        b1c2 = self.leaky_relu(self.b1c2(b1c1))
        b1c3 = self.b1c3(b1c2)
        b1c4 = self.relu(b1c1 + b1c3)
        
        b2c1 = self.leaky_relu(self.b2c1(b1c4))
        b2c2 = self.leaky_relu(self.b2c2(b2c1))
        b2c3 = self.b2c3(b2c2)
        b2c4 = self.relu(b2c1 + b2c3)
        
        b3c1 = self.leaky_relu(self.b3c1(b2c4))
        b3c2 = self.leaky_relu(self.b3c2(b3c1))
        b3c3 = self.dropout(self.b3c3(b3c2))
        b3c3 = self.leaky_relu(b3c3)
        
        # out = self.tanh(self.out(b3c3)) ### want to be between 0-1 so replace? !!!!!!!!!!!!!
        out = self.out(b3c3) ### linear activation
        
        return out

    
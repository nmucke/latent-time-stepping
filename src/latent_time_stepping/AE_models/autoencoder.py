
from typing import Any, Mapping
import torch
from torch import nn
import pdb

class Autoencoder(nn.Module):
    """
    Unsupervised Wasserstein Autoencoder
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x):
        """Encode"""
        return self.encoder(x)
    
    def decode(self, x):
        """Decode"""
        return self.decoder(x)
            
    def forward(self, z):
        """Forward pass"""
        return self.decoder(z)
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
        

    @property
    def device(self):
        return next(self.parameters()).device.type
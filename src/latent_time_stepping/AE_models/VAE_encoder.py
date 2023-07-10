import pdb
import torch
import torch.nn as nn

from latent_time_stepping.AE_models.encoder_decoder import Encoder


class VAEEncoder(nn.Module):
    def __init__(
        self, 
        num_channels: list = None,
        kernel_size: list = None,
        latent_dim: int = 16
        ) -> None:

        super().__init__()
        
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        self.encoder_out_shape = latent_dim*2

        self.encoder = Encoder(
            num_channels=num_channels,
            kernel_size=kernel_size,
            latent_dim=self.encoder_out_shape
        )

        self.mu = nn.Sequential(
            nn.Linear(self.encoder_out_shape, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.logvar = nn.Sequential(
            nn.Linear(self.encoder_out_shape, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps*std

    def forward(self, x):
        x = self.encoder(x)
        
        mu = self.mu(x)
        logvar = self.logvar(x)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
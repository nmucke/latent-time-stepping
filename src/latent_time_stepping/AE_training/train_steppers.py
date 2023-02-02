import torch
from latent_time_stepping.special_loss_functions import MMD
from latent_time_stepping.AE_training.optimizers import Optimizer
import pdb
import torch.nn as nn

import matplotlib.pyplot as plt

class LatentRegressor(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        ) -> None:
        
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = nn.LeakyReLU()

        self.output_activation = nn.Sigmoid()

        self.linear1 = nn.Linear(self.input_size, output_size)
        #self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        #x = self.activation(x)
        #x = self.linear2(x)

        return x#self.output_activation(x)


class BaseTrainStepper():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
    ) -> None:
        pass

    def train_step(self) -> None:
        pass

    def val_step(self) -> None:
        pass


class WAETrainStepper():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        latent_loss_regu: float = 1.0,
        include_time: bool = False,
    ) -> None:
    
        self.model = model
        self.optimizer = optimizer
        self.latent_loss_regu = latent_loss_regu

        self.include_time = include_time
            
        self.device = model.device

        self.reconstruction_loss = 0.0
        self.latent_distribution_loss = 0.0
        self.counter = 0

        self.latent_regressor = LatentRegressor(
            input_size=self.model.encoder.latent_dim,
            hidden_size=16,
            output_size=1,
        ).to(self.device)

        self.latent_regressor_optimizer = torch.optim.Adam(
            self.latent_regressor.parameters(),
            lr=1e-3,
        )
    
    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
    
    def reset_loss(self):
        self.reconstruction_loss = 0.0
        self.latent_distribution_loss = 0.0
        self.counter = 0

    def _reconstruction_loss_function(
        self,
        state: torch.Tensor,
        state_pred: torch.Tensor,
        ) -> torch.Tensor:

        return torch.nn.MSELoss()(state_pred, state)
    
    def _sample_prior(
        self,
        latent_state: torch.Tensor,
        ) -> torch.Tensor:

        """Sample from a standard normal distribution"""
        
        return torch.randn_like(latent_state)

    def _latent_distribution_loss_function(
        self,
        latent_state: torch.Tensor,
        ) -> torch.Tensor:

        z = self._sample_prior(latent_state)

        return MMD(latent_state, z, kernel='multiscale', device=self.device)
    
    def _latent_time_attraction(
        self,
        latent_state: torch.Tensor,
        t: torch.Tensor,
        ) -> torch.Tensor:


        '''
        latent_state = latent_state.to(self.device)
        t = t.to(self.device)
        t = t.unsqueeze(1)

        t = t/1000

        t_pred = self.latent_regressor(latent_state)

        loss = torch.nn.MSELoss()(t_pred, t)

        self.latent_regressor_optimizer.zero_grad()
        loss.backward(retain_graph=True)

        self.latent_regressor_optimizer.step()

        '''
        t = t.to(self.device)
        t = t.unsqueeze(1)
        t = t/1000

        t_dist = torch.abs(t - t.T)
        t_dist = torch.exp(-t_dist)

        latent_dist = torch.cdist(
            latent_state, 
            latent_state,
            compute_mode='donot_use_mm_for_euclid_dist',
            )

        weighted_latent_dist = latent_dist * t_dist

        return weighted_latent_dist.mean()
    
    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        t: torch.Tensor,
        ) -> None:

        self.model.train()

        self.optimizer.zero_grad()

        latent_state = self.model.encoder(state)

        #latent_attraction = self._latent_time_attraction(latent_state, t)
        #self._latent_time_attraction(latent_state, t)

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)

        latent_distribution_loss = \
            self._latent_distribution_loss_function(latent_state)

        loss = \
            reconstruction_loss + self.latent_loss_regu*latent_distribution_loss

        #loss += 1e-4*latent_attraction

        loss.backward()

        self.optimizer.step()

        self.reconstruction_loss += reconstruction_loss.item()
        self.latent_distribution_loss += latent_distribution_loss.item()
        self.counter += 1
        
        return {
            'recon': self.reconstruction_loss/self.counter,
            'latent': self.latent_distribution_loss/self.counter,
        }

    def val_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,        
        ) -> None:

        self.model.eval()

        latent_state = self.model.encoder(state)

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)

        latent_distribution_loss = \
            self._latent_distribution_loss_function(latent_state)

        self.reconstruction_loss += reconstruction_loss.item()
        self.latent_distribution_loss += latent_distribution_loss.item()
        self.counter += 1

        self.val_loss = {
            'recon': self.reconstruction_loss/self.counter,
            'latent': self.latent_distribution_loss/self.counter,
        }


class VAETrainStepper():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        latent_loss_regu: float = 1.0,
    ) -> None:
    
        self.model = model
        self.optimizer = optimizer
        self.latent_loss_regu = latent_loss_regu
            
        self.device = model.device

        self.reconstruction_loss = 0.0
        self.latent_distribution_loss = 0.0
        self.counter = 0
    
    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
    
    def reset_loss(self):
        self.reconstruction_loss = 0.0
        self.latent_distribution_loss = 0.0
        self.counter = 0

    def _reconstruction_loss_function(
        self,
        state: torch.Tensor,
        state_pred: torch.Tensor,
        ) -> torch.Tensor:

        return torch.nn.MSELoss()(state_pred, state)
    
    def _sample_prior(
        self,
        latent_state: torch.Tensor,
        ) -> torch.Tensor:

        """Sample from a standard normal distribution"""
        
        return torch.randn_like(latent_state)

    def _latent_distribution_loss_function(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        ) -> torch.Tensor:

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return kl_divergence.mean()
    
    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        ) -> None:

        self.model.train()

        self.optimizer.zero_grad()

        latent_state, mu, log_var = self.model.encoder(state)

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)

        latent_distribution_loss = \
            self._latent_distribution_loss_function(mu, log_var)

        loss = \
            reconstruction_loss + self.latent_loss_regu*latent_distribution_loss

        loss.backward()

        self.optimizer.step()

        self.reconstruction_loss += reconstruction_loss.item()
        self.latent_distribution_loss += latent_distribution_loss.item()
        self.counter += 1
        
        return {
            'recon': self.reconstruction_loss/self.counter,
            'latent': self.latent_distribution_loss/self.counter,
        }

    def val_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,        
        ) -> None:

        self.model.eval()

        latent_state, mu, log_var = self.model.encoder(state)

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)

        latent_distribution_loss = \
            self._latent_distribution_loss_function(mu, log_var)

        self.reconstruction_loss += reconstruction_loss.item()
        self.latent_distribution_loss += latent_distribution_loss.item()
        self.counter += 1

        self.val_loss = {
            'recon': self.reconstruction_loss/self.counter,
            'latent': self.latent_distribution_loss/self.counter,
        }


        
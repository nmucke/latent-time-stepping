import torch
from latent_time_stepping.special_loss_functions import MMD
from latent_time_stepping.AE_training.optimizers import Optimizer
import pdb

import matplotlib.pyplot as plt


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
        latent_state: torch.Tensor,
        ) -> torch.Tensor:

        z = self._sample_prior(latent_state)

        return MMD(latent_state, z, kernel='multiscale', device=self.device)
    
    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        ) -> None:

        self.model.train()

        self.optimizer.zero_grad()

        latent_state = self.model.encoder(state)

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)

        latent_distribution_loss = \
            self._latent_distribution_loss_function(latent_state)

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


        
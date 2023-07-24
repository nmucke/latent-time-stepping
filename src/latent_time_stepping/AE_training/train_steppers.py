import torch
from latent_time_stepping.special_loss_functions import MMD
from latent_time_stepping.AE_training.optimizers import Optimizer
import pdb
import torch.nn as nn

import matplotlib.pyplot as plt

class LatentRegressor(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        ) -> None:
        
        super().__init__()

        self.input_size = latent_dim
        #self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        return x


class BaseAETrainStepper():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        model_save_path: str,
    ) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.model_save_path = model_save_path

        self.device = model.device

        self.epoch_count = 0
        self.counter = 0

        self.val_loss = None
        self.best_loss = float('inf')

    def _save_model(self,) -> None:
        
        model_save_dict = {
            'encoder': self.model.encoder.state_dict(),
            'decoder': self.model.decoder.state_dict(),
        }
        optimizer_save_dict = {
            'encoder': self.optimizer.encoder.state_dict(),
            'decoder': self.optimizer.decoder.state_dict(),
            'encoder_scheduler': self.optimizer.encoder_scheduler.state_dict(),
            'decoder_scheduler': self.optimizer.decoder_scheduler.state_dict(),
        }

        torch.save(
            {
                'model_state_dict': model_save_dict,
                'optimizer_state_dict': optimizer_save_dict,
            },
            f'{self.model_save_path}/model.pt',
        )

        # save best loss to file
        with open(f'{self.model_save_path}/loss.txt', 'w') as f:
            f.write(str(self.best_loss))

    def _reset_loss(self) -> None:
        raise NotImplementedError

    def start_epoch(self) -> None:
        self.epoch_count += 1
        self.model.train()                                
    
    def end_epoch(self) -> None:
        
        self.optimizer.step_scheduler(self.val_loss['recon'])

        if self.val_loss['recon'] < self.best_loss:
            self.best_loss = self.val_loss['recon']
            self._save_model()
        
        self._reset_loss()

        return self.val_loss

    def train_step(self) -> None:
        pass

    def val_step(self) -> None:
        pass


class WAETrainStepper(BaseAETrainStepper):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        model_save_path: str,
        latent_loss_regu: float = 1.0,
        consistency_loss_regu: float = None,
    ) -> None:
        
        super().__init__(model, optimizer, model_save_path)
    
        self.latent_loss_regu = latent_loss_regu
        self.consistency_loss_regu = consistency_loss_regu

        self.reconstruction_loss = 0.0
        self.latent_distribution_loss = 0.0
        self.consistency_loss = 0.0

        self.recon_loss = torch.nn.MSELoss()

        '''
        self.latent_regressor = LatentRegressor(
            input_size=self.model.encoder.latent_dim,
            hidden_size=16,
            output_size=1,
        ).to(self.device)

        self.latent_regressor_optimizer = torch.optim.Adam(
            self.latent_regressor.parameters(),
            lr=1e-3,
        )
        '''
    
    def _reset_loss(self):
        self.reconstruction_loss = 0.0
        self.latent_distribution_loss = 0.0
        self.consistency_loss = 0.0
        self.counter = 0

    def _reconstruction_loss_function(
        self,
        state: torch.Tensor,
        state_pred: torch.Tensor,
        ) -> torch.Tensor:

        return self.recon_loss(state_pred, state)
    
    
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

        latent_state = latent_state.permute(0, 2, 1)
        latent_state = latent_state.reshape(-1, latent_state.shape[-1])

        z = self._sample_prior(latent_state)

        return MMD(latent_state, z, kernel='multiscale', device=self.device)
    
    def _latent_time_attraction(
        self,
        latent_state: torch.Tensor,
        t: torch.Tensor,
        ) -> torch.Tensor:

        pass

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
        '''
    
    def _latent_consistency_loss_function(
        self,
        latent_state: torch.Tensor,
        latent_pred: torch.Tensor,
        ) -> torch.Tensor:

        latent_loss = self.recon_loss(latent_state, latent_pred) 

        return latent_loss

    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        ) -> None:

        self.counter += 1

        state = state.to(self.device)
        pars = pars.to(self.device)

        self.model.train()

        self.optimizer.zero_grad()

        latent_state = self.model.encoder(state)

        #latent_attraction = self._latent_time_attraction(latent_state, t)
        #self._latent_time_attraction(latent_state, t)

        loss = 0.0

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)
        
        loss += reconstruction_loss

        if self.latent_loss_regu is not None:
            latent_distribution_loss = \
                self._latent_distribution_loss_function(latent_state)

            loss += self.latent_loss_regu * latent_distribution_loss 
            
            self.latent_distribution_loss += latent_distribution_loss.item()
            print_latent_distribution_loss = self.latent_distribution_loss/self.counter

        else:
            print_latent_distribution_loss = None

        if self.consistency_loss_regu is not None:
            latent_pred = self.model.encoder(state_pred)
            consistency_loss = \
                self._latent_consistency_loss_function(latent_state, latent_pred)
            loss += self.consistency_loss_regu * consistency_loss
            
            self.consistency_loss += consistency_loss.item()
            print_consistency_loss = self.consistency_loss/self.counter
        
        else:
            print_consistency_loss = None

        loss.backward()
        self.optimizer.step()

        # chheck if loss is NaN

        self.reconstruction_loss += reconstruction_loss.item()

        output = {
            'recon': self.reconstruction_loss/self.counter,
            'latent': print_latent_distribution_loss,
            'consistency': print_consistency_loss,
        }
        
        return output

    def val_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,        
        ) -> None:

        state = state.to(self.device)
        pars = pars.to(self.device)

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


class AETrainStepper(BaseAETrainStepper):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        model_save_path: str,
        include_time: bool = False,
    ) -> None:
        
        super().__init__(model, optimizer, model_save_path)
    
        self.model = model
        self.optimizer = optimizer

        self.include_time = include_time
            
        self.device = model.device

        self.reconstruction_loss = 0.0

        self.recon_loss = torch.nn.MSELoss()
    
    def _reset_loss(self):
        self.reconstruction_loss = 0.0
        self.latent_distribution_loss = 0.0
        self.counter = 0

    def _reconstruction_loss_function(
        self,
        state: torch.Tensor,
        state_pred: torch.Tensor,
        ) -> torch.Tensor:

        return self.recon_loss(state_pred, state)
    
    def train_step(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        t: torch.Tensor,
        ) -> None:

        state = state.to(self.device)
        pars = pars.to(self.device)

        self.model.train()

        self.optimizer.zero_grad()

        latent_state = self.model.encoder(state)

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)

        loss = reconstruction_loss 

        loss.backward()

        self.optimizer.step()

        self.reconstruction_loss += reconstruction_loss.item()
        self.latent_distribution_loss += 0.0
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

        state = state.to(self.device)
        pars = pars.to(self.device)

        self.model.eval()

        latent_state = self.model.encoder(state)

        state_pred = self.model.decoder(latent_state, pars)

        reconstruction_loss = \
            self._reconstruction_loss_function(state, state_pred)

        self.reconstruction_loss += reconstruction_loss.item()
        self.latent_distribution_loss += 0.0
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


        
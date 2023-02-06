import torch
from latent_time_stepping.special_loss_functions import MMD
from latent_time_stepping.time_stepping_training.optimizers import Optimizer
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


class TimeSteppingTrainStepper():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        teacher_forcing_ratio: float = 0.9,
        teacher_forcing_ratio_reduction: float = 0.9,
        teacher_forcing_ratio_reduction_freq: int= 5
    ) -> None:
    
        self.model = model
        self.optimizer = optimizer
            
        self.device = model.device

        self.loss = 0.0
        self.counter = 0

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_ratio_reduction = teacher_forcing_ratio_reduction
        self.teacher_forcing_ratio_reduction_freq = \
            teacher_forcing_ratio_reduction_freq
        self.teacher_forcing_counter = 0
    
    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
    
    def reset_loss(self):
        self.loss = 0.0
        self.counter = 0
    
    def update_teacher_forcing(self):
        self.teacher_forcing_counter += 1

        if self.teacher_forcing_counter  % self.teacher_forcing_ratio_reduction_freq == 0:
            self.teacher_forcing_ratio *= self.teacher_forcing_ratio_reduction

    def _loss_function(
        self,
        state: torch.Tensor,
        state_pred: torch.Tensor,
        ) -> torch.Tensor:

        return torch.nn.MSELoss()(state_pred, state)
    
    def train_step(
        self,
        input_state: torch.Tensor,
        output_state: torch.Tensor,
        pars: torch.Tensor,
        ) -> None:

        self.model.train()

        self.optimizer.zero_grad()

        state_pred = self.model.masked_prediction(
            x=input_state,
            pars=pars,
        )

        loss = self._loss_function(output_state, state_pred)

        loss.backward()

        self.optimizer.step()

        self.loss += loss.item()
        self.counter += 1
        
        return {
            'loss': self.loss/self.counter
        }

    def val_step(
        self,
        input_state: torch.Tensor,
        output_state: torch.Tensor,
        pars: torch.Tensor,  
        ) -> None:

        self.model.eval()
        output_seq_len = output_state.shape[1]

        #state_pred = self.model.multistep_prediction(
        #    x=input_state,
        #    pars=pars,
        #    output_seq_len=output_seq_len,
        #)
        with torch.no_grad():
            state_pred = self.model.masked_prediction(
                    x=input_state,
                    pars=pars,
                )
            loss = self._loss_function(output_state, state_pred)

        self.loss += loss.item()
        self.counter += 1

        self.val_loss = {
            'loss': self.loss/self.counter,
        }


        
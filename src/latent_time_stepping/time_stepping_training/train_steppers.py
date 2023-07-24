import torch
from latent_time_stepping.special_loss_functions import MMD
from latent_time_stepping.time_stepping_training.optimizers import Optimizer
import pdb

import matplotlib.pyplot as plt



class BaseTimeSteppingTrainStepper():

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
        
        model_save_dict = self.model.state_dict()

        optimizer_save_dict = {
            'optimizer': self.optimizer.optimizer.state_dict(),
            'scheduler': self.optimizer.scheduler.state_dict(),
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
    
    def _update_teacher_forcing(self) -> None:
        pass

    def start_epoch(self) -> None:
        self.epoch_count += 1
        self.model.train()                                
    
    def end_epoch(self) -> None:
        
        self.optimizer.step_scheduler(self.val_loss['loss'])

        if self.val_loss['loss'] < self.best_loss:
            self.best_loss = self.val_loss['loss']
            self._save_model()
        
        self._reset_loss()

        self._update_teacher_forcing()

        return self.val_loss

    def train_step(self) -> None:
        pass

    def val_step(self) -> None:
        pass




class TimeSteppingTrainStepper(BaseTimeSteppingTrainStepper):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        model_save_path: str,
        teacher_forcing_ratio: float = 0.9,
        teacher_forcing_ratio_reduction: float = 0.9,
        teacher_forcing_ratio_reduction_freq: int= 5,
        mixed_precision: bool = False,
    ) -> None:
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            model_save_path=model_save_path,
        )
    
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

        self.mixed_precision = mixed_precision
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _reset_loss(self):
        self.loss = 0.0
        self.counter = 0
    
    def _update_teacher_forcing(self):
        self.teacher_forcing_counter += 1

        if self.teacher_forcing_counter % self.teacher_forcing_ratio_reduction_freq == 0:
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

        input_state = input_state.to(self.device)
        output_state = output_state.to(self.device)
        pars = pars.to(self.device)

        self.model.train()

        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                if torch.rand(1) < self.teacher_forcing_ratio:
                    state_pred = self.model.masked_prediction(
                        input=input_state,
                        output=output_state,
                        pars=pars,
                    )
                else:
                    state_pred = self.model.multistep_prediction(
                        input=input_state,
                        pars=pars,
                        output_seq_len=output_state.shape[-1],
                    )

                loss = self._loss_function(output_state, state_pred)
        else:
            if torch.rand(1) < self.teacher_forcing_ratio:
                state_pred = self.model.masked_prediction(
                    input=input_state,
                    output=output_state,
                    pars=pars,
                )
            else:
                state_pred = self.model.multistep_prediction(
                    input=input_state,
                    pars=pars,
                    output_seq_len=output_state.shape[-1],
                )

            loss = self._loss_function(output_state, state_pred)

        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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

        input_state = input_state.to(self.device)
        output_state = output_state.to(self.device)
        pars = pars.to(self.device)

        self.model.eval()

        with torch.no_grad():
            state_pred = self.model.multistep_prediction(
                input=input_state,
                pars=pars,
                output_seq_len=output_state.shape[-1],
            )
            loss = self._loss_function(output_state, state_pred)

        self.loss += loss.item()
        self.counter += 1

        self.val_loss = {
            'loss': self.loss/self.counter,
        }


        
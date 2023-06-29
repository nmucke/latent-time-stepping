from torch import nn
import torch
from tqdm import tqdm
import pdb
from dataclasses import dataclass


from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import BaseTrainStepper



@dataclass
class EarlyStopping:
    num_non_improving_epochs: int = 0
    best_loss: float = float('inf')
    patience: int = 10

def train(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    model_save_path: str,
    train_stepper: BaseTrainStepper,
    print_progress: bool = True,
    patience: int = None,
    return_val_metrics: bool = False,
) -> None:
    

    if patience is not None:
        early_stopper = EarlyStopping(patience=patience)

    device = train_stepper.device

    for epoch in range(num_epochs):

        if print_progress:
            pbar = tqdm(
                    enumerate(train_dataloader),
                    total=int(len(train_dataloader.dataset)/train_dataloader.batch_size),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        else:
            pbar = enumerate(train_dataloader)

        train_stepper.reset_loss()
        for i, (state, pars, t) in pbar:

            state = state.to(device)
            pars = pars.to(device)

            loss = train_stepper.train_step(state, pars, t)


            if i % 100 == 0:
                pbar.set_postfix(loss)
        
        train_stepper.optimizer.step_scheduler(loss.get('reconstruction_loss', 0))
        #train_stepper.optimizer.step_scheduler()

        train_stepper.reset_loss()
        for i, (state, pars, t) in enumerate(val_dataloader):

            state = state.to(device)
            pars = pars.to(device)

            train_stepper.val_step(state, pars)

        if print_progress:
            for keys, values in train_stepper.val_loss.items():
                print(f'{keys}: {values:.6f}')
                
            print(f'Epoch: {epoch+1}/{num_epochs}')
        
        if patience is not None:
            if train_stepper.val_loss['recon'] < early_stopper.best_loss:
                early_stopper.best_loss = train_stepper.val_loss['recon']
                early_stopper.num_non_improving_epochs = 0

                if model_save_path is not None:
                    train_stepper.save_model(model_save_path)
            else:
                early_stopper.num_non_improving_epochs += 1
                if early_stopper.num_non_improving_epochs >= early_stopper.patience:
                    print('Early stopping')
                    break
                
    if patience is None:
        if model_save_path is not None:
            train_stepper.save_model(model_save_path)
    
    if model_save_path is not None:                                                              
        train_stepper.save_model(model_save_path)

    if return_val_metrics:
        return early_stopper.best_loss
    

'''
class AETrainer():

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizers,
        params: dict,
        model_save_path: str,
        train_stepper: ,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.params = params
        
        self.model_save_path = model_save_path

        self.device = model.device

        self.early_stopper = EarlyStopper(
            patience=params['early_stopping_params']['patience'],
            min_delta=0,
        )

        # Get trainer
        self.train_stepper = create_train_stepper(
            model=model,
            optimizer=optimizer,
            params=params,
        )

    def _train_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader,
    ) -> None:

            pbar = tqdm(
                enumerate(train_dataloader),
                total=int(len(train_dataloader.dataset)/train_dataloader.batch_size),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
            )
            for i, (state, pars) in pbar:

                state = state.to(self.device)
                pars = pars.to(self.device)

                loss = self.train_stepper.train_step(state, pars)

                if i % 100 == 0:
                    pbar.set_postfix(loss)

    def _val_epoch(
        self,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        
        total_loss = {}
        for i, (state, pars) in enumerate(val_dataloader):

            state = state.to(self.device)
            pars = pars.to(self.device)

            loss = self.train_stepper.val_step(state, pars)

            for k in loss.keys():
                total_loss[k] = total_loss.get(k, 0) + loss[k]
        
        print('Validation losses', end=': ')
        for k in total_loss.keys():
            total_loss[k] = total_loss[k]
            print(f'{k}: {total_loss[k]/ len(val_dataloader):.7f}', end=', ')
        print(f'epoch: {self.epoch}', end=' ')
        print()

        return total_loss
    
    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> None:

        for epoch in range(self.params['training_params']['num_epochs']):
            self.epoch = epoch

            self.model.train()
            self._train_epoch(train_dataloader)

            self.model.eval()
            val_loss = self._val_epoch(val_dataloader)

            early_stop, is_best_model = \
                self.early_stopper.early_stop(val_loss['recon_loss'])
            
            if early_stop:
                print('Early stopping')
                break
            if is_best_model:
                torch.save(self.model, self.model_save_path)

            self.train_stepper.scheduler_step()
            
'''


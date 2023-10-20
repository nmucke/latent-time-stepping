from dataclasses import dataclass
from torch import nn
import torch
from tqdm import tqdm
import pdb

from latent_time_stepping.time_stepping_training.train_steppers import BaseTimeSteppingTrainStepper


@dataclass
class EarlyStopping:
    num_non_improving_epochs: int = 0
    best_loss: float = float('inf')
    patience: int = 10


def train(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    train_stepper: BaseTimeSteppingTrainStepper,
    print_progress: bool = True,
    patience: int = None,
) -> None:
    

    if patience is not None:
        early_stopper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):

        # Setup progress bar
        if print_progress:
            pbar = tqdm(
                    enumerate(train_dataloader),
                    total=int(len(train_dataloader.dataset)/train_dataloader.batch_size),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
                )
        else:
            pbar = enumerate(train_dataloader)

        
        #################### Start epoch ####################

        train_stepper.start_epoch()

        # Train
        for i, (input_state, output_state, pars) in pbar:
            
            loss = train_stepper.train_step(
                input_state=input_state,
                output_state=output_state,
                pars=pars,
            )

            if i % 10 == 0:
                pbar.set_postfix(loss)
        
        # Validate
        for i, (input_state, output_state, pars) in enumerate(val_dataloader):
            train_stepper.val_step(
                input_state=input_state,
                output_state=output_state,
                pars=pars,
            )

        val_loss = train_stepper.end_epoch()
        
        #################### End epoch ####################
        
        # Print val loss
        if print_progress:
            for keys, values in val_loss.items():
                print(f'{keys}: {values:.6f}', end=', ')
                
            print(f'Epoch: {epoch+1}/{num_epochs}')
        
        # Early stopping
        if patience is not None:
            if train_stepper.val_loss['loss'] < early_stopper.best_loss:
                early_stopper.best_loss = train_stepper.val_loss['loss']
                early_stopper.num_non_improving_epochs = 0
            else:
                early_stopper.num_non_improving_epochs += 1
                if early_stopper.num_non_improving_epochs >= early_stopper.patience:
                    print('Early stopping')
                    break
    
'''
def train(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    model_save_path: str,
    train_stepper: str,
    print_progress: bool = True,
    patience: int = None
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
        for i, (input_state, output_state, pars) in pbar:

            input_state = input_state.to(device)
            output_state = output_state.to(device)
            pars = pars.to(device)

            loss = train_stepper.train_step(
                input_state=input_state,
                output_state=output_state,
                pars=pars,
                )

            if i % 100 == 0:
                pbar.set_postfix(loss)
        
        train_stepper.optimizer.step_scheduler(loss.get('reconstruction_loss', 0))
        train_stepper.update_teacher_forcing()

        train_stepper.reset_loss()
        for i, (input_state, output_state, pars) in enumerate(val_dataloader):

            input_state = input_state.to(device)
            output_state = output_state.to(device)
            pars = pars.to(device)

            train_stepper.val_step(
                input_state=input_state,
                output_state=output_state,
                pars=pars,
            )

        if print_progress:
            for keys, values in train_stepper.val_loss.items():
                print(f'{keys}: {values:.6f}')
            print(f'Teacher forcing ratio: {train_stepper.teacher_forcing_ratio:.2f}')
            print(f'Learning rate: {train_stepper.optimizer.scheduler.get_lr()[0]:.6f}')
            print(f'Epoch: {epoch+1}/{num_epochs}')
        
        if patience is not None:
            if train_stepper.val_loss['loss'] < early_stopper.best_loss:
                early_stopper.best_loss = train_stepper.val_loss['loss']
                early_stopper.num_non_improving_epochs = 0
                train_stepper.save_model(model_save_path)
            else:
                early_stopper.num_non_improving_epochs += 1
                if early_stopper.num_non_improving_epochs >= early_stopper.patience:
                    print('Early stopping')
                    break
                
    if patience is None:
        train_stepper.save_model(model_save_path)
'''

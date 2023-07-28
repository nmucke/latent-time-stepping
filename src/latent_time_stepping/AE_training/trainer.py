import numpy as np
from torch import nn
import torch
from tqdm import tqdm
import pdb
from dataclasses import dataclass

from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import BaseAETrainStepper

@dataclass
class EarlyStopping:
    num_non_improving_epochs: int = 0
    best_loss: float = float('inf')
    patience: int = 10

def train(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    train_stepper: BaseAETrainStepper,
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
        for i, (state, pars) in pbar:
            
            # pick N random integers from 0 to len(train_dataloader)
             
            idx = np.random.randint(0, state.shape[-1], size=state.shape[-1]//8)
            state = state[:, :, :, idx]

            loss = train_stepper.train_step(state, pars)

            if i % 10 == 0:
                pbar.set_postfix(loss)
        
        # Validate
        for i, (state, pars) in enumerate(val_dataloader):
            idx = np.random.randint(0, state.shape[-1], size=state.shape[-1]//16)
            state = state[:, :, :, idx]

            train_stepper.val_step(state, pars)

        val_loss = train_stepper.end_epoch()
        
        #################### End epoch ####################
        
        # Print val loss
        if print_progress:
            for keys, values in val_loss.items():
                print(f'{keys}: {values:.6f}', end=', ')
                
            print(f'Epoch: {epoch+1}/{num_epochs}')
        
        # Early stopping
        if patience is not None:
            if train_stepper.val_loss['recon'] < early_stopper.best_loss:
                early_stopper.best_loss = train_stepper.val_loss['recon']
                early_stopper.num_non_improving_epochs = 0
            else:
                early_stopper.num_non_improving_epochs += 1
                if early_stopper.num_non_improving_epochs >= early_stopper.patience:
                    print('Early stopping')
                    break
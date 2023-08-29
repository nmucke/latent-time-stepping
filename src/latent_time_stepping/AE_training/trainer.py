import numpy as np
from torch import nn
import torch
from tqdm import tqdm
import pdb
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time

from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import BaseAETrainStepper

from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation

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
    patience: int = None,
    print_level: int = 1,
) -> None:
    

    if patience is not None:
        early_stopper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):

        # Setup progress bar
        if print_level==2:
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
        t1 = time.time()
        for i, (state, pars) in pbar:
            
            # pick N random integers from 0 to len(train_dataloader)
            #idx = np.random.randint(0, state.shape[-1], size=state.shape[-1]//8)
            #state = state[:, :, :, idx]

            '''
            lal = savgol_filter(state, 15, 1, axis=-2)
            #lal = savgol_filter(state, 20, 3, axis=-2)
            x = np.linspace(0, 5000, state.shape[-2])

            fig, ax = plt.subplots()
            xdata, ydata, ydata_1 = [], [], []
            ln, = ax.plot([], [], lw=3, animated=True)
            ln_1, = ax.plot([], [], lw=3, animated=True)

            def init():
                ax.set_xlim(0, 5000)
                ax.set_ylim(state.min(), state.max())
                #ax.set_ylim(0.1, 0.9)
                #ax.set_ylim(18, 35)
                return ln,

            def update(frame):
                xdata.append(x)
                ydata.append(state[0, 1, :, frame])
                ydata_1.append(lal[0, 1, :, frame])
                ln.set_data(x, state[0, 1, :, frame])
                ln_1.set_data(x, lal[0, 1, :, frame])
                return ln, ln_1,

            ani = FuncAnimation(
                fig,
                update,
                frames=np.arange(0, state.shape[-1], 1),
                init_func=init, 
                blit=True,
                interval=10,
                )
            ani.save('pipeflow_holdup.gif', fps=30)
            plt.show()
            '''

            loss = train_stepper.train_step(state, pars)

            if print_level == 2:
                if i % 10 == 0:
                    pbar.set_postfix(loss)
        # Validate
        for i, (state, pars) in enumerate(val_dataloader):

            train_stepper.val_step(state, pars)

        val_loss = train_stepper.end_epoch()
        
        t2 = time.time()
        #################### End epoch ####################
        
        # Print val loss
        if print_level==1 or print_level==2:
            for keys, values in val_loss.items():
                print(f'{keys}: {values:.6f}', end=', ')
                
            print(f'Epoch: {epoch+1}/{num_epochs}, time: {t2-t1:.2f} s')
        
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

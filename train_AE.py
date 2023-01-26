import pdb
from matplotlib import pyplot as plt
import yaml
from yaml.loader import SafeLoader
import torch

from latent_time_stepping.AE_models.autoencoder import UnsupervisedWassersteinAE

from latent_time_stepping.AE_models.encoder_decoder import (
    Decoder, 
    Encoder
)
from latent_time_stepping.datasets.AE_dataset import get_AE_dataloader
from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import WAETrainStepper
from latent_time_stepping.AE_training.trainer import train

torch.set_default_dtype(torch.float32)

MODEL_TYPE = "WAE"

config_path = f"configs/{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=SafeLoader)

STATE_PATH = 'data/processed_data/training_data/states.pt'
PARS_PATH = 'data/processed_data/training_data/pars.pt'

TRAIN_SAMPLE_IDS = range(2500)
VAL_SAMPLE_IDS = range(2500, 3000)

state = torch.load(STATE_PATH)
pars = torch.load(PARS_PATH)

train_state = state[TRAIN_SAMPLE_IDS]
train_pars = pars[TRAIN_SAMPLE_IDS]

val_state = state[VAL_SAMPLE_IDS]
val_pars = pars[VAL_SAMPLE_IDS]


MODEL_SAVE_PATH = f"trained_models/autoencoders/{MODEL_TYPE}.pt"

CUDA = True
DEVICE = torch.device('cuda' if CUDA else 'cpu')

def main():

    encoder = Encoder(**config['model_args']['encoder'])
    decoder = Decoder(**config['model_args']['decoder'])

    model = UnsupervisedWassersteinAE(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to(DEVICE)
    
    optimizer = Optimizer(
        model=model,
        args=config['optimizer_args'],
    )

    train_stepper = WAETrainStepper(
        model=model,
        optimizer=optimizer,
        **config['train_stepper_args'],
    )

    train_dataloader = get_AE_dataloader(
        state=train_state,
        pars=train_pars,
        **config['dataloader_args']
    )
    val_dataloader = get_AE_dataloader(
        state=val_state,
        pars=val_pars,
        **config['dataloader_args']
    )
    
    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model_save_path=MODEL_SAVE_PATH,
        train_stepper=train_stepper,
        print_progress=True,
        **config['train_args'],
    )

if __name__ == "__main__":
    
    main()

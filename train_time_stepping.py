import pdb
import yaml
from yaml.loader import SafeLoader
import torch
import matplotlib.pyplot as plt
from latent_time_stepping.time_stepping_training.optimizers import Optimizer
from latent_time_stepping.time_stepping_models.parameter_encoder import ParameterEncoder

from latent_time_stepping.time_stepping_models.time_stepping_model import TimeSteppingModel

from latent_time_stepping.datasets.time_stepping_dataset import get_time_stepping_dataloader
from latent_time_stepping.time_stepping_training.train_steppers import TimeSteppingTrainStepper

from latent_time_stepping.time_stepping_training.trainer import train

torch.set_default_dtype(torch.float32)

MODEL_TYPE = "time_stepping"

config_path = f"configs/{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=SafeLoader)
    
LATENT_STATE_PATH = 'data/processed_data/training_data/latent_states.pt'
PARS_PATH = 'data/processed_data/training_data/pars.pt'

TRAIN_SAMPLE_IDS = range(850)
VAL_SAMPLE_IDS = range(850, 1000)

latent_state = torch.load(LATENT_STATE_PATH)
pars = torch.load(PARS_PATH)

train_state = latent_state[TRAIN_SAMPLE_IDS]
train_pars = pars[TRAIN_SAMPLE_IDS]

val_state = latent_state[VAL_SAMPLE_IDS]
val_pars = pars[VAL_SAMPLE_IDS]

MODEL_SAVE_PATH = f"trained_models/time_steppers/{MODEL_TYPE}.pt"

CUDA = True
DEVICE = torch.device('cuda' if CUDA else 'cpu')

def main():

    train_dataloader = get_time_stepping_dataloader(
        state=train_state,
        pars=train_pars,
        **config['dataloader_args']
    )
    val_dataloader = get_time_stepping_dataloader(
        state=val_state,
        pars=val_pars,
        **config['dataloader_args']
    )
    pars_encoder = ParameterEncoder(**config['model_args']['parameter_encoder_args'])
    model = TimeSteppingModel(
        pars_encoder=pars_encoder,
        **config['model_args']['time_stepping_decoder'],
    )
    model = model.to(DEVICE)

    optimizer = Optimizer(
        model=model,
        args=config['optimizer_args'],
    )
    
    train_stepper = TimeSteppingTrainStepper(
        model=model,
        optimizer=optimizer,
        **config['train_stepper_args'],
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

import pdb
from matplotlib import pyplot as plt
import yaml
from yaml.loader import SafeLoader
import torch

from latent_time_stepping.utils import create_directory
from latent_time_stepping.AE_models.VAE_encoder import VAEEncoder
from latent_time_stepping.AE_models.autoencoder import Autoencoder

from latent_time_stepping.AE_models.encoder_decoder import (
    Decoder, 
    Encoder
)
from latent_time_stepping.datasets.AE_dataset import AEDataset, get_AE_dataloader
from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import (
    AETrainStepper, 
    VAETrainStepper, 
    WAETrainStepper
)
from latent_time_stepping.AE_training.trainer import train

torch.set_default_dtype(torch.float32)

MODEL_TYPE = "WAE"
CUDA = True

PHASE = "single"

ORACLE_PATH = None#f'{PHASE}_phase/train'

dataset = AEDataset(
    oracle_path=ORACLE_PATH,
    include_time=True,
    num_skip_steps=4,
)

STATE_PATH = 'data/processed_data/training_data/states.pt'
PARS_PATH = 'data/processed_data/training_data/pars.pt'

TRAIN_SAMPLE_IDS = range(300)
VAL_SAMPLE_IDS = range(300, 400)

MODEL_SAVE_PATH = f"trained_models/autoencoders/{MODEL_TYPE}"

config_path = f"configs/{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=SafeLoader)

BUCKET_NAME = "bucket-20230222-1753"

state = torch.load(STATE_PATH)
pars = torch.load(PARS_PATH)

train_state = state[TRAIN_SAMPLE_IDS]
train_pars = pars[TRAIN_SAMPLE_IDS]

val_state = state[VAL_SAMPLE_IDS]
val_pars = pars[VAL_SAMPLE_IDS]

create_directory(MODEL_SAVE_PATH)

# Save config file
with open(f'{MODEL_SAVE_PATH}/config.yml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

if CUDA:
    DEVICE = torch.device('cuda' if CUDA else 'cpu')
else:
    DEVICE = torch.device('cpu')

def main():

    if MODEL_TYPE == "VAE":
        encoder = VAEEncoder(**config['model_args']['encoder'])
    elif MODEL_TYPE == "WAE":
        encoder = Encoder(**config['model_args']['encoder'])
    elif MODEL_TYPE == "AE":
        encoder = Encoder(**config['model_args']['encoder'])

    decoder = Decoder(**config['model_args']['decoder'])

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to(DEVICE)
    
    optimizer = Optimizer(
        model=model,
        args=config['optimizer_args'],
    )

    if MODEL_TYPE == "VAE":
        train_stepper = VAETrainStepper(
            model=model,
            optimizer=optimizer,
            **config['train_stepper_args'],
        )
    elif MODEL_TYPE == "WAE":
        train_stepper = WAETrainStepper(
            model=model,
            optimizer=optimizer,
            **config['train_stepper_args'],
        )
    elif MODEL_TYPE == "AE":
        train_stepper = AETrainStepper(
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
        model_save_path=f'{MODEL_SAVE_PATH}/model.pt',
        train_stepper=train_stepper,
        print_progress=True,
        **config['train_args'],
    )

if __name__ == "__main__":
    
    main()

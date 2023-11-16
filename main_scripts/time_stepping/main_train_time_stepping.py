import pdb
import numpy as np
import yaml
from yaml.loader import SafeLoader
import torch
import matplotlib.pyplot as plt
from latent_time_stepping.time_stepping_models.neural_ODE import NODETimeSteppingModel
from latent_time_stepping.time_stepping_training.optimizers import Optimizer
from latent_time_stepping.time_stepping_models.parameter_encoder import ParameterEncoder

from latent_time_stepping.time_stepping_models.time_stepping_model import TimeSteppingModel

from latent_time_stepping.datasets.time_stepping_dataset import TimeSteppingDataset
from latent_time_stepping.time_stepping_training.train_steppers import TimeSteppingTrainStepper

from latent_time_stepping.time_stepping_training.trainer import train
from latent_time_stepping.utils import create_directory

torch.set_default_dtype(torch.float32)

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

CONTINUE_TRAINING = False

MODEL_TYPE = "transformer"

PHASE = "burgers"

config_path = f"configs/neural_networks/{PHASE}_phase_{MODEL_TYPE}.yml"
with open(config_path) as f:
    config = yaml.load(f, Loader=SafeLoader)
    
LOCAL_OR_ORACLE = 'local'

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/latent_data/train'

LOCAL_LOAD_PATH = f'data/{PHASE}_phase/latent_data/train'

MODEL_SAVE_PATH = f"trained_models/time_steppers/{PHASE}_phase_{MODEL_TYPE}"
create_directory(MODEL_SAVE_PATH)
with open(f'{MODEL_SAVE_PATH}/config.yml', 'w') as f:
    yaml.dump(config, f)

DEVICE = 'cuda'

if PHASE == 'single':
    num_skip_steps = 4
    NUM_PARS = 2
    NUM_SAMPLES = 2500
    NUM_STATES = 2
    LOAD_MODEL_FROM_ORACLE = False
elif PHASE == 'multi':
    num_skip_steps = 10
    NUM_PARS = 2
    NUM_STATES = 3
    NUM_SAMPLES = 5000
    LOAD_MODEL_FROM_ORACLE = True
elif PHASE == 'lorenz':
    num_skip_steps = 1
    NUM_SAMPLES = 1000
    NUM_STATES = 1
    NUM_PARS = 1
    LOAD_MODEL_FROM_ORACLE = True
elif PHASE == 'wave':
    num_skip_steps = 1
    NUM_SAMPLES = 210
    NUM_STATES = 2
    NUM_PARS = 1
    LOAD_MODEL_FROM_ORACLE = False
elif PHASE == 'burgers':
    num_skip_steps = 1
    NUM_SAMPLES = 1000
    NUM_STATES = 1
    NUM_PARS = 1
    LOAD_MODEL_FROM_ORACLE = False

SAMPLE_IDS = range(NUM_SAMPLES)

def main():

    dataset = TimeSteppingDataset(
        local_path=LOCAL_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        filter=True if PHASE == 'multi' else False,
        **config['dataset_args'],
    )
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(0.8*len(dataset)), int(0.2*len(dataset))]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        **config['dataloader_args'],
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        **config['dataloader_args'],
    )

    if MODEL_TYPE == 'transformer':
        if config['model_args']['parameter_encoder_args'] is not None:
            pars_encoder = ParameterEncoder(**config['model_args']['parameter_encoder_args'])
        else:
            pars_encoder = None
        model = TimeSteppingModel(
            pars_encoder=pars_encoder,
            **config['model_args']['time_stepping_decoder'],
        )
    elif MODEL_TYPE == 'NODE':
        model = NODETimeSteppingModel(
            **config['model_args'],
        )

    model = model.to(DEVICE)

    optimizer = Optimizer(
        model=model,
        args=config['optimizer_args'],
    )

    if CONTINUE_TRAINING:
        state_dict = torch.load(f'{MODEL_SAVE_PATH}/model.pt', map_location=DEVICE)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    #model = torch.compile(model)

    train_stepper = TimeSteppingTrainStepper(
        model=model,
        optimizer=optimizer,
        model_save_path=MODEL_SAVE_PATH,
        **config['train_stepper_args'],
    )

    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_stepper=train_stepper,
        print_progress=True,
        **config['train_args'],
    )

if __name__ == "__main__":
    main()

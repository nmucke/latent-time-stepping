
import os
import pdb
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from latent_time_stepping.AE_models.VAE_encoder import VAEEncoder
from latent_time_stepping.AE_models.autoencoder import Autoencoder
from latent_time_stepping.AE_models.encoder_decoder import Decoder, Encoder
from latent_time_stepping.time_stepping_models.FNO_time_stepping_model import FNOTimeSteppingModel
from latent_time_stepping.time_stepping_models.neural_ODE import NODETimeSteppingModel
from latent_time_stepping.time_stepping_models.parameter_encoder import ParameterEncoder
from latent_time_stepping.time_stepping_models.time_stepping_model import TimeSteppingModel

def create_directory(directory):
    """
    Creates a directory if it doesn't exist
    :param directory: The directory to create
    :return: None
    """

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)


def load_trained_AE_model(
    model_load_path = None, 
    state_dict = None,
    config = None,
    model_type = 'WAE', 
    device = 'cpu'
    ):

    if model_load_path is not None:
        state_dict = torch.load(f'{model_load_path}/model.pt', map_location=device)

        with open(f'{model_load_path}/config.yml') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    if model_type == "VAE":
        encoder = VAEEncoder(**config['model_args']['encoder'])
    elif model_type == "WAE":
        encoder = Encoder(**config['model_args']['encoder'])
    elif model_type == "AE":
        encoder = Encoder(**config['model_args']['encoder'])

    decoder = Decoder(**config['model_args']['decoder'])

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to(device)
    model.load_state_dict(state_dict['model_state_dict'])

    return model

def load_trained_time_stepping_model(
    model_load_path = None, 
    model_type = 'transformer',
    state_dict = None,
    config = None,
    device = 'cpu',
):
    if model_load_path is not None:
        state_dict = torch.load(f'{model_load_path}/model.pt', map_location=device)

        with open(f'{model_load_path}/config.yml') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    if model_type == 'FNO':
        model = FNOTimeSteppingModel(
            **config['model_args'],
        )
    elif model_type == 'NODE':
        model = NODETimeSteppingModel(
            **config['model_args'],
        )
    else:
        if config['model_args']['parameter_encoder_args'] is not None:
            pars_encoder = ParameterEncoder(**config['model_args']['parameter_encoder_args'])
        else:
            pars_encoder = None
        model = TimeSteppingModel(
            pars_encoder=pars_encoder,
            **config['model_args']['time_stepping_decoder'],
        )
    
    model = model.to(device)
    model.load_state_dict(state_dict['model_state_dict'])

    return model
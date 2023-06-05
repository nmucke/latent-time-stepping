import pdb
from matplotlib import pyplot as plt
import yaml
from yaml.loader import SafeLoader
import torch

from latent_time_stepping.AE_models.VAE_encoder import VAEEncoder

from latent_time_stepping.AE_models.autoencoder import Autoencoder

from latent_time_stepping.AE_models.encoder_decoder import (
    Decoder, 
    Encoder
)
from latent_time_stepping.datasets.AE_dataset import get_AE_dataloader
from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import (
    AETrainStepper, 
    VAETrainStepper, 
    WAETrainStepper
)
from latent_time_stepping.AE_training.trainer import train

torch.set_default_dtype(torch.float32)


def fitness_function(config, train_loader, test_loader):

    
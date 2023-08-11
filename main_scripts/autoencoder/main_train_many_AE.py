import os
import pdb
from matplotlib import pyplot as plt
import numpy as np
import yaml
from yaml.loader import SafeLoader
import torch
import ray
from latent_time_stepping.oracle import ObjectStorageClientWrapper

from latent_time_stepping.utils import create_directory
from latent_time_stepping.AE_models.VAE_encoder import VAEEncoder
from latent_time_stepping.AE_models.autoencoder import Autoencoder

from latent_time_stepping.AE_models.encoder_decoder import (
    Decoder, 
    Encoder
)
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.AE_training.optimizers import Optimizer
from latent_time_stepping.AE_training.train_steppers import (
    AETrainStepper, 
    VAETrainStepper, 
    WAETrainStepper
)
from latent_time_stepping.AE_training.trainer import train

torch.set_default_dtype(torch.float32)

@ray.remote(num_cpus=1)
def train_remote(
    latent_dim,
):
    PHASE = "single"
    
    CUDA = False
    if CUDA:
        DEVICE = torch.device('cuda' if CUDA else 'cpu')
    else:
        DEVICE = torch.device('cpu')

    MODEL_SAVE_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE"

    NUM_SAMPLES = 20
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2

    SAMPLE_IDS = range(NUM_SAMPLES)

    config_path = f"configs/neural_networks/{PHASE}_phase_WAE.yml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/train'

    PREPROCESSOR_PATH = f'{PHASE}_phase/preprocessor.pkl'
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    
    object_storage_client = ObjectStorageClientWrapper(
        bucket_name='trained_models'
    )

    preprocessor = object_storage_client.get_preprocessor(
        source_path=PREPROCESSOR_PATH
    )
    
    dataset = AEDataset(
        oracle_path=ORACLE_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        load_entire_dataset=False,
        num_random_idx_divisor=4,
        preprocessor=preprocessor,
        #num_skip_steps=4
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(TRAIN_RATIO*len(dataset)), int(VAL_RATIO*len(dataset))]
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        **config['dataloader_args'],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        **config['dataloader_args'],
    )

    config['model_args']['encoder']['latent_dim'] = latent_dim
    config['model_args']['decoder']['latent_dim'] = latent_dim

    oracle_model_save_path = f'{PHASE}_phase/autoencoders/WAE_{latent_dim}'

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

    train_stepper = WAETrainStepper(
        model=model,
        optimizer=optimizer,
        model_save_path=MODEL_SAVE_PATH,
        oracle_path=oracle_model_save_path,
        **config['train_stepper_args'],
    )
    
    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_stepper=train_stepper,
        print_progress=True,
        **config['train_args'],
    )

    return 0

def main():
    for latent_dim in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]:

        _ = train_remote.remote(
            latent_dim=latent_dim,
        )       

        ray.get(_)

if __name__ == "__main__":
    
    ray.init(num_cpus=10)
    main()
    ray.shutdown()

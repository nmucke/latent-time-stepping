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

@ray.remote(num_cpus=8, num_gpus=1)
def train_remote(
    transposed,
    resnet,
    num_channels,
    num_layers,
    phase
):
    
    
    CONTINUE_TRAINING = False
    PHASE = phase

    latent_dim = 4 if PHASE == "single" else 8

    CUDA = True
    if CUDA:
        DEVICE = torch.device('cuda' if CUDA else 'cpu')
    else:
        DEVICE = torch.device('cpu')

    if PHASE == "single":
        NUM_SAMPLES = 2500
    else:
        NUM_SAMPLES = 5000
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2

    SAMPLE_IDS = range(NUM_SAMPLES)

    config_path = f"configs/neural_networks/{PHASE}_phase_WAE.yml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/train'
    LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/train'

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
        #oracle_path=ORACLE_LOAD_PATH,
        local_path=LOCAL_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        load_entire_dataset=False,
        num_random_idx_divisor=1 if PHASE == "single" else 2,
        preprocessor=preprocessor,
        num_skip_steps=4 if PHASE == "single" else 5,
        #states_to_include=(1,2) if PHASE == "multi" else None,
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

    config['model_args']['decoder']['transposed'] = transposed

    config['model_args']['encoder']['resnet'] = resnet
    config['model_args']['decoder']['resnet'] = resnet

    if PHASE == "single":
        config['model_args']['decoder']['num_channels'] = [num_channels//(2**i) for i in range(0, num_layers)]
        config['model_args']['decoder']['num_channels'].append(2)
        config['model_args']['encoder']['num_channels'] = config['model_args']['decoder']['num_channels'][::-1]
    elif PHASE == "multi":
        config['model_args']['decoder']['num_channels'] = [num_channels//(2**i) for i in range(0, num_layers)]
        config['model_args']['decoder']['num_channels'].append(3)
        config['model_args']['encoder']['num_channels'] = config['model_args']['decoder']['num_channels'][::-1]


    oracle_model_save_path = f'{PHASE}_phase/autoencoders/WAE_{latent_dim}_layers_{num_layers}_channels_{num_channels}'
    MODEL_SAVE_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE_{latent_dim}_layers_{num_layers}_channels_{num_channels}"

    if transposed:
        oracle_model_save_path += "_transposed"
        MODEL_SAVE_PATH += "_transposed"
    if resnet:
        oracle_model_save_path += "_resnet"
        MODEL_SAVE_PATH += "_resnet"

    create_directory(MODEL_SAVE_PATH)

    # Save config file
    with open(f'{MODEL_SAVE_PATH}/config.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

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

    if CONTINUE_TRAINING:
        state_dict, _ = object_storage_client.get_model(
            source_path=oracle_model_save_path,
            device=DEVICE,
        )
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

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

    print(f'Finished training with latent_dim={latent_dim}')

    return 0

def main():
    out = []

    transposed_list = [False, True]
    resnet_list = [False, True]
    num_channels_list = [128]

    PHASE = "single"

    if PHASE == "single":
        num_layers_list = [5, 6]
    elif PHASE == "multi":
        num_layers_list = [6, 7]

    for transposed in transposed_list:
        for resnet in resnet_list:
            for num_channels in num_channels_list:
                for num_layers in num_layers_list:
                    
                    _ = train_remote.remote(
                        transposed=transposed,
                        resnet=resnet,
                        num_channels=num_channels,
                        num_layers=num_layers,
                        phase=PHASE,
                    )   

                    out.append(_)

    ray.get(out)

if __name__ == "__main__":
    
    ray.init(num_cpus=32, num_gpus=4)
    main()
    ray.shutdown()

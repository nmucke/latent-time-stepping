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

torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

torch.set_default_dtype(torch.float32)

@ray.remote(num_cpus=4)#, num_gpus=1)
def train_remote(
    config_path: str,
    oracle_model_save_path: str,
):  
    
    CONTINUE_TRAINING = False
    PHASE = 'multi'

    if PHASE == 'single':
        num_skip_steps = 4
    elif PHASE == 'multi':
        num_skip_steps = 10
        NUM_SAMPLES = 5000
    elif PHASE == 'lorenz':
        num_skip_steps = 5
    elif PHASE == 'burgers':
        num_skip_steps = 1

    CUDA = True
    if CUDA:
        DEVICE = torch.device('cuda' if CUDA else 'cpu')
    else:
        DEVICE = torch.device('cpu')


    TRAIN_RATIO = 0.8

    FULL_SAMPLE_IDS = range(NUM_SAMPLES)
    num_train_samples = int(TRAIN_RATIO*NUM_SAMPLES)
    num_val_samples = NUM_SAMPLES - num_train_samples

    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/train'
    PREPROCESSOR_PATH = f'{PHASE}_phase/preprocessor.pkl'
    
    object_storage_client = ObjectStorageClientWrapper(
        bucket_name='trained_models'
    )

    preprocessor = object_storage_client.get_preprocessor(
        source_path=PREPROCESSOR_PATH
    )
    
    dataset = AEDataset(
        #oracle_path=ORACLE_LOAD_PATH,
        local_path=LOCAL_LOAD_PATH,
        sample_ids=FULL_SAMPLE_IDS,
        load_entire_dataset=False,
        num_random_idx_divisor=None if PHASE == "multi" else None,
        preprocessor=preprocessor,
        num_skip_steps=num_skip_steps,
        filter=True if PHASE == "multi" else False,
        #states_to_include=(1,2) if PHASE == "multi" else None,
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [num_train_samples, num_val_samples]
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        **config['dataloader_args'],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        **config['dataloader_args'],
    )

    MODEL_SAVE_PATH = oracle_model_save_path


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
        print_level=1,
        **config['train_args'],
    )

    return 0

def main():
    out = []

    conf_path_list = [
        'multi_phase_conv_WAE.yml',
        'multi_phase_AE_no_reg.yml'
    ]

    model_name_list = [
        'WAE_conv',
        'AE_no_reg'
    ]

    oracle_model_save_path =  f'multi_phase/autoencoders/'

    for conf, model_name in zip(conf_path_list, model_name_list):

        oracle_model_save_path = f'multi_phase/autoencoders/{model_name}'

        config_path = f"configs/neural_networks/{conf}"

        _ = train_remote.remote(
            config_path=config_path,
            oracle_model_save_path=oracle_model_save_path,
        )   

        out.append(_)

    ray.get(out)

if __name__ == "__main__":
    ray.init(num_cpus=8)#, num_gpus=4)
    main()
    ray.shutdown()

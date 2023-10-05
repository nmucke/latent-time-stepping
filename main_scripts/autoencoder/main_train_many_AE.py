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

@ray.remote(num_cpus=16, num_gpus=1)
def train_remote(
    latent_dim,
    transposed,
    resnet,
    num_channels,
    num_layers,
    phase,
    embedding_dim,
    consistency_loss_regu,
    latent_loss_regu,
    num_transformer_layers,
    vit,
):  
    
    CONTINUE_TRAINING = False
    PHASE = phase

    if PHASE == 'single':
        num_skip_steps = 4
    elif PHASE == 'multi':
        num_skip_steps = 10
    elif PHASE == 'lorenz':
        num_skip_steps = 5

    #latent_dim = 4 if PHASE == "single" else 8

    CUDA = True
    if CUDA:
        DEVICE = torch.device('cuda' if CUDA else 'cpu')
    else:
        DEVICE = torch.device('cpu')

    if PHASE == "single":
        NUM_SAMPLES = 2500
    elif PHASE == "lorenz":
        NUM_SAMPLES = 3000
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
        num_random_idx_divisor=None if PHASE == "multi" else None,
        preprocessor=preprocessor,
        num_skip_steps=num_skip_steps,
        filter=True if PHASE == "multi" else False,
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

    #config['model_args']['decoder']['transposed'] = transposed

    #config['model_args']['encoder']['resnet'] = resnet
    #config['model_args']['decoder']['resnet'] = resnet

    config['model_args']['decoder']['transposed'] = transposed

    config['model_args']['decoder']['resnet'] = resnet
    config['model_args']['encoder']['resnet'] = resnet

    config['model_args']['decoder']['vit'] = vit
    config['model_args']['encoder']['vit'] = vit



    if PHASE == "single":
        config['model_args']['decoder']['num_channels'] = [num_channels//(2**i) for i in range(0, num_layers)]
        config['model_args']['decoder']['num_channels'].append(2)
        config['model_args']['encoder']['num_channels'] = config['model_args']['decoder']['num_channels'][::-1]
        config['model_args']['encoder']['embedding_dim'] = [embedding_dim for _ in range(5)]
        config['model_args']['decoder']['embedding_dim'] = [embedding_dim for _ in range(5)]
    elif PHASE == "multi":
        config['model_args']['decoder']['num_channels'] = [num_channels//(2**i) for i in range(0, num_layers)]
        config['model_args']['decoder']['num_channels'].append(3)
        config['model_args']['encoder']['num_channels'] = config['model_args']['decoder']['num_channels'][::-1]
        config['model_args']['encoder']['embedding_dim'] = [embedding_dim for _ in range(6)]
        config['model_args']['decoder']['embedding_dim'] = [embedding_dim for _ in range(6)]
    else:
        config['model_args']['decoder']['num_channels'] = [num_channels//(2**i) for i in range(0, num_layers)]
        config['model_args']['decoder']['num_channels'].append(1)
        config['model_args']['encoder']['num_channels'] = config['model_args']['decoder']['num_channels'][::-1]
        config['model_args']['encoder']['embedding_dim'] = [embedding_dim for _ in range(5)]
        config['model_args']['decoder']['embedding_dim'] = [embedding_dim for _ in range(5)]


    config['train_stepper_args']['latent_loss_regu'] = latent_loss_regu
    config['train_stepper_args']['consistency_loss_regu'] = consistency_loss_regu


    config['model_args']['encoder']['num_transformer_layers'] = num_transformer_layers
    config['model_args']['decoder']['num_transformer_layers'] = num_transformer_layers

    #oracle_model_save_path = f'{PHASE}_phase/autoencoders/WAE_{latent_dim}_layers_{num_layers}_channels_{num_channels}'
    #MODEL_SAVE_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE_{latent_dim}_layers_{num_layers}_channels_{num_channels}"
    oracle_model_save_path = f'{PHASE}_phase/autoencoders/WAE_{latent_dim}'
    oracle_model_save_path += f"_latent_{latent_loss_regu}"
    oracle_model_save_path += f"_consistency_{consistency_loss_regu}"
    oracle_model_save_path += f"_channels_{num_channels}"
    oracle_model_save_path += f"_layers_{num_layers}"
    if vit:
        oracle_model_save_path += f"_trans_layers_{num_transformer_layers}"
        oracle_model_save_path += f"_embedding_{embedding_dim}"

    if transposed:
        oracle_model_save_path += "_transposed"
    if resnet:
        oracle_model_save_path += "_resnet"
    if vit:
        oracle_model_save_path += "_vit"

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

    print(f'Finished training with latent_dim={latent_dim}')

    return 0

def main():
    out = []

    transposed_list = [False]
    resnet_list = [False]
    vit_list = [True]

    num_channels_list = [128, 64]

    embedding_dim_list = [64]
    latent_loss_regu_list = [1e-3, 1e-4]
    consistency_loss_regu_list = [1e-2, 1e-3]

    latent_dim_list = [8, 12, 16]

    num_transformer_layers_list = [1, 2]

    PHASE = "lorenz"


    if PHASE == "single":
        num_layers_list = [5, 6]
    elif PHASE == "multi":
        num_layers_list = [6, 7]
    else:
        num_layers_list = [5, 6]


    for transposed in transposed_list:
        for resnet in resnet_list:
            for num_channels in num_channels_list:
                for num_layers in num_layers_list:
                    for embedding_dim in embedding_dim_list:
                        for latent_loss_regu in latent_loss_regu_list:
                            for consistency_loss_regu in consistency_loss_regu_list:
                                for latent_dim in latent_dim_list:
                                    for num_transformer_layers in num_transformer_layers_list:
                                        for vit in vit_list:

                                            _ = train_remote.remote(
                                                latent_dim=latent_dim,
                                                transposed=transposed,
                                                resnet=resnet,
                                                num_channels=num_channels,
                                                num_layers=num_layers,
                                                phase=PHASE,
                                                embedding_dim=embedding_dim,
                                                latent_loss_regu=latent_loss_regu,
                                                consistency_loss_regu=consistency_loss_regu,
                                                num_transformer_layers=num_transformer_layers,
                                                vit=vit,
                                            )   

                                            out.append(_)

    ray.get(out)

if __name__ == "__main__":
    
    ray.init(num_cpus=64, num_gpus=4)
    main()
    ray.shutdown()

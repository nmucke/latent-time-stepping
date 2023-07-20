import os
import torch
from tqdm import tqdm
import numpy as np
import pdb
import pickle

import matplotlib.pyplot as plt
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.oracle.oracle import ObjectStorageClientWrapper

from latent_time_stepping.preprocessor import Preprocessor
from latent_time_stepping.utils import create_directory

torch.set_default_dtype(torch.float32)

NUM_SAMPLES = 2500
SAMPLE_IDS = range(NUM_SAMPLES)

NUM_SKIP_STEPS = 5
END_TIME_INDEX = 25000

LOCAL_OR_ORACLE = 'local'
PHASE = 'single'
TRAIN_OR_TEST = 'train'

NUM_WORKERS = 64
BATCH_SIZE = 40

ORACLE_LOAD_PATH = f'{PHASE}_phase/{TRAIN_OR_TEST}'
ORACLE_SAVE_PATH = f'{PHASE}_phase/processed_data/{TRAIN_OR_TEST}'
BUCKET_NAME = "bucket-20230222-1753"

LOCAL_LOAD_PATH = f'data/raw_data/training_data'
LOCAL_SAVE_PATH = f'data/processed_data/{TRAIN_OR_TEST}'
create_directory(LOCAL_SAVE_PATH)

TRAINED_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors'
create_directory(TRAINED_PREPROCESSOR_SAVE_PATH)

TRAINED_PREPROCESSOR_SAVE_PATH += '/single_phase_preprocessor.pt'

if PHASE == 'single':
    NUM_STATES = 2
elif PHASE == 'multi':
    NUM_STATES = 3

def main():

    ############### Train the preprocessor #####################
    if LOCAL_OR_ORACLE == 'oracle':
        dataset = AEDataset(
            oracle_path=ORACLE_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            sample_ids=SAMPLE_IDS,
            end_time_index=END_TIME_INDEX,
        )
    else:
        dataset = AEDataset(
            local_path=LOCAL_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            sample_ids=SAMPLE_IDS,
            end_time_index=END_TIME_INDEX,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    preprocessor = Preprocessor(num_states=NUM_STATES, num_pars=2)

    # Fit the preprocessor
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (state, pars) in pbar:
        preprocessor.partial_fit_state(state)
        preprocessor.partial_fit_pars(pars)

    # Save the preprocessor   
    torch.save(preprocessor, TRAINED_PREPROCESSOR_SAVE_PATH)

    ############### Save processed data #####################

    if LOCAL_OR_ORACLE == 'oracle':
        dataset = AEDataset(
            oracle_path=ORACLE_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            sample_ids=SAMPLE_IDS,
            end_time_index=END_TIME_INDEX,
            preprocessor=preprocessor,
        )
    else:
        dataset = AEDataset(
            local_path=LOCAL_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            sample_ids=SAMPLE_IDS,
            end_time_index=END_TIME_INDEX,
            preprocessor=preprocessor,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Transform the data
    processed_states = []
    processed_pars = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (state, pars) in pbar:
        processed_states.append(state)
        processed_pars.append(pars)
        
    processed_states = torch.cat(processed_states, dim=0)
    processed_pars = torch.cat(processed_pars, dim=0)

    processed_states = processed_states.numpy()
    processed_pars = processed_pars.numpy()


    # Save the processed data
    if LOCAL_OR_ORACLE == 'oracle':

        object_storage_client = ObjectStorageClientWrapper(BUCKET_NAME)

        object_storage_client.put_object(
            data=processed_states,
            destination_path=f'{ORACLE_SAVE_PATH}/states.npz',
        )
        object_storage_client.put_object(
            data=processed_pars,
            destination_path=f'{ORACLE_SAVE_PATH}/pars.npz',
        )

    elif LOCAL_OR_ORACLE == 'local':

        np.savez_compressed(
            f'{LOCAL_SAVE_PATH}/states.npz',
            data=processed_states
        )
        np.savez_compressed(
            f'{LOCAL_SAVE_PATH}/pars.npz',
            data=processed_pars,
    )

if __name__ == "__main__":
    
    main()

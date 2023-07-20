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

torch.set_default_dtype(torch.float32)

LOCAL_OR_ORACLE = 'oracle'
PHASE = 'single'
TRAIN_OR_TEST = 'train'

NUM_SKIP_STEPS = 4

ORACLE_LOAD_PATH = f'{PHASE}_phase/{TRAIN_OR_TEST}'
ORACLE_SAVE_PATH = f'{PHASE}_phase/processed_data/{TRAIN_OR_TEST}'
BUCKET_NAME = "bucket-20230222-1753"

LOCAL_LOAD_PATH = f'data/raw_data/training_data'
LOCAL_SAVE_PATH = f'data/processed_data/{TRAIN_OR_TEST}'
if LOCAL_OR_ORACLE == 'local':
    if not os.path.exists(LOCAL_SAVE_PATH):
        os.makedirs(LOCAL_SAVE_PATH)

TRAINED_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors'

# Check if path exists
if not os.path.exists(TRAINED_PREPROCESSOR_SAVE_PATH):
    os.makedirs(TRAINED_PREPROCESSOR_SAVE_PATH)

TRAINED_PREPROCESSOR_SAVE_PATH += '/single_phase_preprocessor.pt'

def main():

    ############### Train the preprocessor #####################
    if LOCAL_OR_ORACLE == 'oracle':
        dataset = AEDataset(
            oracle_path=ORACLE_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            num_samples=5000,
        )
    else:
        dataset = AEDataset(
            local_path=LOCAL_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            num_samples=5000,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=32,
    )

    preprocessor = Preprocessor(num_states=2, num_pars=2)

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
            num_samples=5000,
            preprocessor=preprocessor,
        )
    else:
        dataset = AEDataset(
            local_path=LOCAL_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            num_samples=5000,
            preprocessor=preprocessor,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=32,
    )

    # Transform the data
    processed_states = []
    processed_pars = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (state, pars) in pbar:
        processed_states.append(state)
        processed_pars.append(pars)
    
    processed_states = torch.stack(processed_states)
    processed_pars = torch.stack(processed_pars)

    torch.save(processed_states, f'{LOCAL_SAVE_PATH}/states.pt')
    torch.save(processed_pars, f'{LOCAL_SAVE_PATH}/pars.pt')

    # Save the processed data
    if LOCAL_OR_ORACLE == 'oracle':

        object_storage_client = ObjectStorageClientWrapper(BUCKET_NAME)

        object_storage_client.put_object(
            destination_path=f'{ORACLE_SAVE_PATH}/states.pt',
        )
        object_storage_client.put_object(
            destination_path=f'{ORACLE_SAVE_PATH}/pars.pt',
        )

    elif LOCAL_OR_ORACLE == 'local':

        torch.save(processed_states, f'{LOCAL_SAVE_PATH}/states.pt')
        torch.save(processed_pars, f'{LOCAL_SAVE_PATH}/pars.pt')
    

if __name__ == "__main__":
    
    main()

import os
import torch
from tqdm import tqdm
import numpy as np
import pdb
import pickle

import matplotlib.pyplot as plt
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.oracle import ObjectStorageClientWrapper

from latent_time_stepping.preprocessor import Preprocessor
from latent_time_stepping.utils import create_directory

torch.set_default_dtype(torch.float32)


NUM_SKIP_STEPS = 1
END_TIME_INDEX = 10000

LOCAL_OR_ORACLE = 'local'
PHASE = 'lorenz'
TRAIN_OR_TEST = 'train'

NUM_WORKERS = 8
BATCH_SIZE = 64

ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/{TRAIN_OR_TEST}'
BUCKET_NAME = "bucket-20230222-1753"

LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/train'

TRAINED_PREPROCESSOR_SAVE_PATH = 'trained_preprocessors'
create_directory(TRAINED_PREPROCESSOR_SAVE_PATH)

TRAINED_PREPROCESSOR_SAVE_PATH += f'/{PHASE}_phase_preprocessor.pkl'


if PHASE == 'single':
    NUM_STATES = 2
    num_skip_steps = 4
    NUM_SAMPLES = 2500
elif PHASE == 'multi':
    NUM_STATES = 3
    num_skip_steps = 10
    NUM_SAMPLES = 5000
elif PHASE == 'lorenz':
    NUM_STATES = 1
    num_skip_steps = 1
    NUM_SAMPLES = 3000
elif PHASE == 'wave':
    NUM_STATES = 2
    num_skip_steps = 1
    NUM_SAMPLES = 210

SAMPLE_IDS = range(NUM_SAMPLES)

def main():

    ############### Train the preprocessor #####################
    if LOCAL_OR_ORACLE == 'oracle':
        dataset = AEDataset(
            oracle_path=ORACLE_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            sample_ids=SAMPLE_IDS,
            #end_time_index=END_TIME_INDEX,
        )
    else:
        dataset = AEDataset(
            local_path=LOCAL_LOAD_PATH,
            num_skip_steps=NUM_SKIP_STEPS,
            sample_ids=SAMPLE_IDS,
            #end_time_index=END_TIME_INDEX,
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
    with open(TRAINED_PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)

    object_storage_client = ObjectStorageClientWrapper(
        bucket_name='trained_models'
    )

    object_storage_client.put_preprocessor(
        source_path=TRAINED_PREPROCESSOR_SAVE_PATH,
        destination_path=f'{PHASE}_phase/preprocessor.pkl',
    )

    '''

            
        
    #torch.save(preprocessor, TRAINED_PREPROCESSOR_SAVE_PATH)

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

        object_storage_client.put_numpy_object(
            data=processed_states,
            destination_path=f'{ORACLE_SAVE_PATH}/states.npz',
        )
        object_storage_client.put_numpy_object(
            data=processed_pars,
            destination_path=f'{ORACLE_SAVE_PATH}/pars.npz',
        )


        object_storage_client = ObjectStorageClientWrapper(
            bucket_name='trained_models'
        )

        object_storage_client.put_preprocessor(
            source_path=TRAINED_PREPROCESSOR_SAVE_PATH,
            destination_path=f'{PHASE}_phase/preprocessor.pkl',
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
    '''
if __name__ == "__main__":
    
    main()

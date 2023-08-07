import os
import pdb
import pickle
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader
import torch
from latent_time_stepping.oracle import ObjectStorageClientWrapper

from latent_time_stepping.datasets.AE_dataset import AEDataset


PHASE = "single"

TRAIN_OR_TEST = 'train'

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/{TRAIN_OR_TEST}'

ORACLE_SAVE_PATH = f'{PHASE}_phase/raw_data/{TRAIN_OR_TEST}'

NUM_SAMPLES = 2000

TRAIN_SAMPLE_IDS = range(NUM_SAMPLES)

PREPROCESSOR_PATH = f'trained_preprocessors/{PHASE}_phase_preprocessor.pkl'
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

SAVE_FOLDER = f'data/{PHASE}_phase/processed_data/{TRAIN_OR_TEST}'

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(f'{SAVE_FOLDER}/state')
    os.makedirs(f'{SAVE_FOLDER}/pars')
    
def main():

    dataset = AEDataset(
        oracle_path=ORACLE_LOAD_PATH,
        sample_ids=TRAIN_SAMPLE_IDS,
        load_entire_dataset=False,
        #preprocessor=preprocessor,
        num_skip_steps=1,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=64,
    )
    bucket_name = "bucket-20230222-1753"

    object_storage_client = ObjectStorageClientWrapper(bucket_name)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (state, pars) in pbar:
        
        state = state.squeeze(0).numpy()
        pars = pars.squeeze(0).numpy()

        '''
        pdb.set_trace()

        object_storage_client.put_numpy_object(
            destination_path=f'{ORACLE_SAVE_PATH}/state/sample_{i}.npz',
            data=state
        )
        object_storage_client.put_numpy_object(
            destination_path=f'{ORACLE_SAVE_PATH}/pars/sample_{i}.npz',
            data=pars
        )
        '''



        #np.savez_compressed(f'{SAVE_FOLDER}/state/sample_{i}', data=state)
        #np.savez_compressed(f'{SAVE_FOLDER}/pars/sample_{i}', data=pars)
        

if __name__ == "__main__":
    main()
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

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/train'
ORACLE_SAVE_PATH = f'{PHASE}_phase/raw_data/train'

NUM_SAMPLES = 2000

TRAIN_SAMPLE_IDS = range(NUM_SAMPLES)


PREPROCESSOR_PATH = f'trained_preprocessors/{PHASE}_phase_preprocessor.pkl'
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)
    
def main():

    dataset = AEDataset(
        oracle_path=ORACLE_LOAD_PATH,
        sample_ids=TRAIN_SAMPLE_IDS,
        load_entire_dataset=False,
        #preprocessor=preprocessor,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=30,
    )

    object_storage_client_wrapper = ObjectStorageClientWrapper(
        bucket_name="bucket-20230222-1753"
    )

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (state, pars) in pbar:
        x = i


if __name__ == "__main__":
    main()
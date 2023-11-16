import pdb
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.oracle import ObjectStorageClientWrapper

from latent_time_stepping.utils import (
    load_trained_AE_model, 
    load_trained_time_stepping_model
)
torch.set_default_dtype(torch.float32)



NUM_SKIP_STEPS = 5

DEVICE = 'cpu'

PHASE = "multi"
AE_MODEL_TYPE = "WAE"
TIME_STEPPING_MODEL_TYPE = "FNO"

AE_model_path = None#f"trained_models/autoencoders/{PHASE}_phase_{AE_MODEL_TYPE}"

time_stepping_model_path = f"trained_models/time_steppers/{PHASE}_phase_{TIME_STEPPING_MODEL_TYPE}"

PREPROCESSOR_PATH = f'trained_preprocessors/{PHASE}_phase_preprocessor.pkl'

BUCKET_NAME = "trained_models"
ORACLE_AE_SAVE_PATH = f'{PHASE}_phase/autoencoders/{AE_MODEL_TYPE}'
ORACLE_TIME_STEPPING_SAVE_PATH = f'{PHASE}_phase/time_steppers/{TIME_STEPPING_MODEL_TYPE}'
ORACLE_PREPROCESSOR_SAVE_PATH = f'{PHASE}_phase/preprocessor.pkl'

def main():
    object_storage_client = ObjectStorageClientWrapper(BUCKET_NAME)

    '''
    ##### Upload preprocessor #####
    object_storage_client.put_preprocessor(
        source_path=PREPROCESSOR_PATH,
        destination_path=ORACLE_PREPROCESSOR_SAVE_PATH,
    )
    print("Preprocessor uploaded to oracle")
    '''

    ##### Upload AE model #####
    if AE_model_path is not None:
        object_storage_client.put_model(
            source_path=AE_model_path,
            destination_path=ORACLE_AE_SAVE_PATH,
        )

        print("AE model uploaded to oracle")

    ##### Upload time stepping model #####
    if time_stepping_model_path is not None:
        object_storage_client.put_model(
            source_path=time_stepping_model_path,
            destination_path=ORACLE_TIME_STEPPING_SAVE_PATH,
        )
        
        print("Time stepping model uploaded to oracle")

    ##### load preprocessor #####
    preprocessor = object_storage_client.get_preprocessor(
        source_path=ORACLE_PREPROCESSOR_SAVE_PATH,
    )

    ##### load AE model #####
    if AE_model_path is not None:
        state_dict, config = object_storage_client.get_model(
            source_path=ORACLE_AE_SAVE_PATH,
        )
        AE_model = load_trained_AE_model(
            state_dict=state_dict,
            config=config,
            model_type=AE_MODEL_TYPE,
            device=DEVICE,
            )                                                  

    ##### load time stepping model #####
    if time_stepping_model_path is not None:
        state_dict, config = object_storage_client.get_model(
            source_path=ORACLE_TIME_STEPPING_SAVE_PATH,
        )
        time_stepping_model = load_trained_time_stepping_model(
            state_dict=state_dict,
            config=config,
            device=DEVICE,
            )


if __name__ == '__main__':
    main()

        

import pdb
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.oracle import ObjectStorageClientWrapper

from latent_time_stepping.utils import create_directory, load_trained_AE_model
torch.set_default_dtype(torch.float32)

DEVICE = 'cuda'

PHASE = "single"
MODEL_TYPE = "WAE"

LATENT_DIM = 4

if PHASE == "single":
    NUM_STATES = 2
elif PHASE == "multi":
    NUM_STATES = 3


LOAD_MODEL_FROM_ORACLE = False

MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}"
ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_{LATENT_DIM}'

object_storage_client = ObjectStorageClientWrapper(
    bucket_name='trained_models'
)

state_dict, config = object_storage_client.get_model(
    source_path=ORACLE_MODEL_LOAD_PATH,
    device=DEVICE,
)
#model.load_state_dict(state_dict['model_state_dict'])
model = load_trained_AE_model(
    model_load_path=MODEL_LOAD_PATH if not LOAD_MODEL_FROM_ORACLE else None,
    state_dict=state_dict if LOAD_MODEL_FROM_ORACLE else None,
    config=config,
    model_type=MODEL_TYPE,
    device=DEVICE,
)




NUM_SAMPLES = 2500
SAMPLE_IDS = range(NUM_SAMPLES)

NUM_PARS = 2

LOCAL_OR_ORACLE = 'local'

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/train'
ORACLE_SAVE_PATH = f'{PHASE}_phase/latent_data/train'

LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/train'
LOCAL_SAVE_PATH = f'data/{PHASE}_phase/latent_data/train'



object_storage_client = ObjectStorageClientWrapper(
    bucket_name='trained_models'
)
PREPROCESSOR_PATH = f'{PHASE}_phase/preprocessor.pkl'
preprocessor = object_storage_client.get_preprocessor(
    source_path=PREPROCESSOR_PATH
)


if LOCAL_OR_ORACLE == 'oracle':
    dataset = AEDataset(
        oracle_path=ORACLE_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        load_entire_dataset=False,
        num_skip_steps=4,
        preprocessor=preprocessor,
    )
elif LOCAL_OR_ORACLE == 'local':

    create_directory(LOCAL_SAVE_PATH)

    dataset = AEDataset(
        local_path=LOCAL_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        load_entire_dataset=False,
        num_skip_steps=4,
        preprocessor=preprocessor,
    )
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

def main():

    state, _ = dataset.__getitem__(0)

    latent_state = torch.zeros((
        NUM_SAMPLES, 
        model.encoder.latent_dim,
        state.shape[-1],
        ))
    
    pars = torch.zeros((
        NUM_SAMPLES,
        NUM_PARS,
        ))
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (state, _pars) in pbar:

        state = state.to(DEVICE)
        
        latent_state[i] = model.encode(state).detach().cpu()[0]
        pars[i] = _pars.detach().cpu()[0]  


    latent_state = latent_state.numpy()
    pars = pars.numpy()

    # Save the processed data
    if LOCAL_OR_ORACLE == 'oracle':

        object_storage_client = ObjectStorageClientWrapper(BUCKET_NAME)

        object_storage_client.put_numpy_object(
            data=latent_state,
            destination_path=f'{ORACLE_SAVE_PATH}/states.npz',
        )
        object_storage_client.put_numpy_object(
            data=pars,
            destination_path=f'{ORACLE_SAVE_PATH}/pars.npz',
        )

    elif LOCAL_OR_ORACLE == 'local':

        np.savez_compressed(
            f'{LOCAL_SAVE_PATH}/states.npz',
            data=latent_state
        )
        np.savez_compressed(
            f'{LOCAL_SAVE_PATH}/pars.npz',
            data=pars,
    )

if __name__ == "__main__":
    
    main()

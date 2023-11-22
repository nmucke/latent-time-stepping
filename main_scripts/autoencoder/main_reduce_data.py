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

PHASE = "multi"
#MODEL_TYPE = "WAE"
MODEL_TYPE = "AE_no_reg"
#MODEL_TYPE = "WAE_conv"

TRANSPOSED = True
RESNET = False
NUM_CHANNELS = 256 if PHASE == 'multi' else 128
NUM_LAYERS = 6

LATENT_DIM = 16


if PHASE == 'single':
    num_skip_steps = 4
    NUM_PARS = 2
    NUM_SAMPLES = 2500
    NUM_STATES = 2
    LOAD_MODEL_FROM_ORACLE = False
elif PHASE == 'multi':
    num_skip_steps = 10
    NUM_PARS = 2
    NUM_STATES = 3
    NUM_SAMPLES = 5000
    LOAD_MODEL_FROM_ORACLE = True
elif PHASE == 'lorenz':
    num_skip_steps = 1
    NUM_SAMPLES = 3000
    NUM_STATES = 1
    NUM_PARS = 1
    LOAD_MODEL_FROM_ORACLE = False
elif PHASE == 'wave':
    num_skip_steps = 1
    NUM_SAMPLES = 210
    NUM_STATES = 2
    NUM_PARS = 1
    LOAD_MODEL_FROM_ORACLE = False
elif PHASE == 'burgers':
    num_skip_steps = 1
    NUM_SAMPLES = 1024
    NUM_STATES = 1
    NUM_PARS = 1
    LOAD_MODEL_FROM_ORACLE = True

MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}"

if PHASE == 'multi':

    if MODEL_TYPE == "AE_no_reg" or MODEL_TYPE == "WAE_conv":
        ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/{MODEL_TYPE}'
    else:
        ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_8_latent_0.0001_consistency_0.01_channels_128_layers_6_trans_layers_2_embedding_64_vit' #'multi_phase/autoencoders/WAE_8_latent_0.001_consistency_0.01_channels_128_layers_6_trans_layers_1_embedding_64_vit'

elif PHASE == 'lorenz':
    MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}"
    #ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_16_latent_0.001_consistency_0.01_channels_64_layers_3_trans_layers_1_embedding_64_vit'
elif PHASE == 'single':
    MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}_vit_conv_{LATENT_DIM}_1_trans_layer"
    ORACLE_MODEL_LOAD_PATH = None
elif PHASE == 'wave':
    MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}"
    ORACLE_MODEL_LOAD_PATH = None
elif PHASE == 'burgers':
    MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}"
    ORACLE_MODEL_LOAD_PATH = 'burgers_phase/autoencoders/WAE_8_latent_0.0001_consistency_0.001_channels_64_layers_4_trans_layers_2_embedding_64_vit'

object_storage_client = ObjectStorageClientWrapper(
    bucket_name='trained_models'
)

if LOAD_MODEL_FROM_ORACLE:
    state_dict, config = object_storage_client.get_model(
        source_path=ORACLE_MODEL_LOAD_PATH,
        device=DEVICE,
    )

model_type = MODEL_TYPE
if PHASE == 'multi':
    if MODEL_TYPE == "AE_no_reg":
        model_type = "WAE"
    elif MODEL_TYPE == "WAE_conv":
        model_type = "WAE"
#model.load_state_dict(state_dict['model_state_dict'])
model = load_trained_AE_model(
    model_load_path=MODEL_LOAD_PATH if not LOAD_MODEL_FROM_ORACLE else None,
    state_dict=state_dict if LOAD_MODEL_FROM_ORACLE else None,
    config=config if LOAD_MODEL_FROM_ORACLE else None,
    model_type=model_type,
    device=DEVICE,
)
model.eval()


SAMPLE_IDS = range(NUM_SAMPLES)


LOCAL_OR_ORACLE = 'local'

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/train'
ORACLE_SAVE_PATH = f'{PHASE}_phase/latent_data/train'

LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/train'
LOCAL_SAVE_PATH = f'data/{PHASE}_phase/latent_data/train'
if PHASE == 'multi':
    if MODEL_TYPE == "AE_no_reg":
        LOCAL_SAVE_PATH = f'data/{PHASE}_phase/latent_data_AE/train'
    elif MODEL_TYPE == "WAE_conv":
        LOCAL_SAVE_PATH = f'data/{PHASE}_phase/latent_data_conv/train'

object_storage_client = ObjectStorageClientWrapper(
    bucket_name='trained_models'
)
PREPROCESSOR_PATH = f'{PHASE}_phase/preprocessor.pkl'
preprocessor = object_storage_client.get_preprocessor(
    source_path=PREPROCESSOR_PATH
)

if LOCAL_OR_ORACLE == 'local':
    create_directory(LOCAL_SAVE_PATH)

    if PHASE == 'multi':
        LOCAL_LOAD_PATH = f'../../../../../scratch2/ntm/data/{PHASE}_phase/raw_data/train'
    else:
        LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/train'

dataset = AEDataset(
    oracle_path=ORACLE_LOAD_PATH if LOCAL_OR_ORACLE == 'oracle' else None,                                                          
    local_path=LOCAL_LOAD_PATH if LOCAL_OR_ORACLE == 'local' else None,
    sample_ids=SAMPLE_IDS,
    preprocessor=preprocessor,
    num_skip_steps=num_skip_steps,
    end_time_index=None,
    filter=True if PHASE == 'multi' else False,
)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=30,
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

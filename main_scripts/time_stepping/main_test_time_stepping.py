import pdb
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from scipy.signal import savgol_filter
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.oracle import ObjectStorageClientWrapper

from latent_time_stepping.utils import load_trained_AE_model, load_trained_time_stepping_model
torch.set_default_dtype(torch.float32)



DEVICE = 'cpu'

PHASE = "multi"
AE_MODEL_TYPE = "WAE"
TIME_STEPPING_MODEL_TYPE = "transformer"
LOAD_MODEL_FROM_ORACLE = True

NUM_SKIP_STEPS = 4 if PHASE == 'single' else 10

LATENT_DIM = 8 if PHASE == 'multi' else 4

if PHASE == "single":
    NUM_STATES = 2
elif PHASE == "multi":
    NUM_STATES = 3

MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE"
if PHASE == 'multi':
    ORACLE_MODEL_LOAD_PATH = 'multi_phase/autoencoders/WAE_8_latent_0.0001_consistency_0.01_channels_128_layers_6_trans_layers_2_embedding_64_vit'#'multi_phase/autoencoders/WAE_8_latent_0.001_consistency_0.01_channels_128_layers_6_trans_layers_1_embedding_64_vit'
else:
    #ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_{LATENT_DIM}_layers_{NUM_LAYERS}_channels_{NUM_CHANNELS}'
    ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_{LATENT_DIM}_consistency'
    MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE_vit_conv_{LATENT_DIM}_1_trans_layer"

object_storage_client = ObjectStorageClientWrapper(
    bucket_name='trained_models'
)

state_dict, config = object_storage_client.get_model(
    source_path=ORACLE_MODEL_LOAD_PATH,
    device=DEVICE,
)
#model.load_state_dict(state_dict['model_state_dict'])
AE = load_trained_AE_model(
    model_load_path=MODEL_LOAD_PATH if not LOAD_MODEL_FROM_ORACLE else None,
    state_dict=state_dict if LOAD_MODEL_FROM_ORACLE else None,
    config=config,
    model_type='WAE',
    device=DEVICE,
)
AE.eval()

time_stepping_model_path = f"trained_models/time_steppers/{PHASE}_phase_{TIME_STEPPING_MODEL_TYPE}"
time_stepper = load_trained_time_stepping_model(
    model_load_path=time_stepping_model_path,
    device=DEVICE,
    model_type=TIME_STEPPING_MODEL_TYPE,
)
if TIME_STEPPING_MODEL_TYPE == 'transformer':
    input_seq_len = 64#time_stepper.input_seq_len
else:
    input_seq_len = 32

time_stepper.eval()


"""
PREPROCESSOR_PATH = f'trained_preprocessors/{PHASE}_phase_preprocessor.pkl'
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)
"""

object_storage_client = ObjectStorageClientWrapper(
    bucket_name='trained_models'
)
PREPROCESSOR_PATH = f'{PHASE}_phase/preprocessor.pkl'
preprocessor = object_storage_client.get_preprocessor(
    source_path=PREPROCESSOR_PATH
)

LOCAL_OR_ORACLE = 'oracle'

LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/training_data'

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/test'

SAMPLE_IDS = range(4, 5)

dataset = AEDataset(
    oracle_path=ORACLE_LOAD_PATH if LOCAL_OR_ORACLE == 'oracle' else None,                                                          
    local_path=LOCAL_LOAD_PATH if LOCAL_OR_ORACLE == 'local' else None,
    sample_ids=SAMPLE_IDS,
    preprocessor=preprocessor,
    num_skip_steps=4 if PHASE == 'single' else 10,
    end_time_index=None,
    filter=True if PHASE == 'multi' else False,
    #states_to_include=(1,2) if PHASE == "multi" else None,
)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

def main():

    num_steps = 1000

    for i, (state, pars) in enumerate(dataloader):
        state = state.to(DEVICE)
        pars = pars.to(DEVICE)

        if TIME_STEPPING_MODEL_TYPE == "FNO":
            pred_recon_state = time_stepper.multistep_prediction(
                input=state[0:1, :, :, 0:input_seq_len],
                pars=pars,
                output_seq_len=num_steps,
                )
            
        else:

            latent_state = AE.encode(state)

            #latent_state = savgol_filter(latent_state.detach().numpy(), 10, 1, axis=-1)
            #latent_state = torch.tensor(latent_state, dtype=torch.float32)

            pred_latent_state = time_stepper.multistep_prediction(
                latent_state[:, :, 0:input_seq_len],
                pars,
                output_seq_len=num_steps,
            )
            pred_latent_state = torch.cat(
                [latent_state[:, :, 0:input_seq_len], pred_latent_state],
                dim=2
            )
                
            pred_recon_state = AE.decode(pred_latent_state, pars)

        pred_recon_state = preprocessor.inverse_transform_state(pred_recon_state, ensemble=True)

        state = preprocessor.inverse_transform_state(state, ensemble=True)
        pred_recon_state = pred_recon_state.detach()
    
    if TIME_STEPPING_MODEL_TYPE != 'FNO':
        latent_state = latent_state.detach().numpy()
        pred_latent_state = pred_latent_state.detach().numpy()
    state = state.detach().numpy()
    pred_recon_state = pred_recon_state.detach().numpy()

    num_latent_to_plot = 4
    plt.figure(figsize=(15,5))

    if TIME_STEPPING_MODEL_TYPE != 'FNO':
        plt.subplot(1, 1, 1)
        plt.plot(latent_state[0, 0, :num_steps], label='latent state', color='tab:blue')
        for i in range(1, num_latent_to_plot):
            plt.plot(latent_state[0, i, :num_steps], color='tab:blue')

        plt.plot(pred_latent_state[0, 0, :num_steps], label='pred latent state', color='tab:orange')
        for i in range(1, num_latent_to_plot):
            plt.plot(pred_latent_state[0, i, :num_steps], color='tab:orange') 
        plt.legend()
        plt.show()

    time_step_to_plot_1 = 100
    time_step_to_plot_2 = 500
    time_step_to_plot_3 = -1
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(state[0, 0, :, time_step_to_plot_1], label=f'HF, t={time_step_to_plot_1}', color='tab:blue')
    plt.plot(pred_recon_state[0, 0, :, time_step_to_plot_1], label=f'NN, t={time_step_to_plot_1}', color='tab:orange')
    plt.plot(state[0, 0, :, time_step_to_plot_2], color='tab:blue', label=f'HF, t={time_step_to_plot_2}')
    plt.plot(pred_recon_state[0, 0, :, time_step_to_plot_2], color='tab:orange')
    #plt.plot(state[0, 0, :, time_step_to_plot_3], color='tab:blue', label=f'HF, t={time_step_to_plot_3}')
    #plt.plot(pred_recon_state[0, 0, :, time_step_to_plot_3], color='tab:orange')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(state[0, 1, :, time_step_to_plot_1], color='tab:blue', label=f'HF, t={time_step_to_plot_1}')
    plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_1], label='pred state', color='tab:orange')
    plt.plot(state[0, 1, :, time_step_to_plot_2], color='tab:blue', label=f'HF, t={time_step_to_plot_2}')
    plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_2], color='tab:orange')
    #plt.plot(state[0, 1, :, time_step_to_plot_3], color='tab:blue', label=f'HF, t={time_step_to_plot_3}')
    #plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_3], color='tab:orange')
    plt.legend()

    if PHASE == 'multi':
        plt.subplot(1, 3, 3)
        plt.plot(state[0, 2, :, time_step_to_plot_1], color='tab:blue', label=f'HF, t={time_step_to_plot_1}')
        plt.plot(pred_recon_state[0, 2, :, time_step_to_plot_1], label='pred state', color='tab:orange')
        plt.plot(state[0, 2, :, time_step_to_plot_2], color='tab:blue', label=f'HF, t={time_step_to_plot_2}')
        plt.plot(pred_recon_state[0, 2, :, time_step_to_plot_2], color='tab:orange')
        #plt.plot(state[0, 2, :, time_step_to_plot_3], color='tab:blue', label=f'HF, t={time_step_to_plot_3}')
        #plt.plot(pred_recon_state[0, 2, :, time_step_to_plot_3], color='tab:orange')
        plt.legend()
    plt.show()


if __name__ == "__main__":
    
    main()

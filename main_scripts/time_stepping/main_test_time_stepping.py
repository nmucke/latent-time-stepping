import pdb
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.oracle import ObjectStorageClientWrapper

from latent_time_stepping.utils import load_trained_AE_model, load_trained_time_stepping_model
torch.set_default_dtype(torch.float32)


NUM_SKIP_STEPS = 4

DEVICE = 'cpu'

PHASE = "single"
AE_MODEL_TYPE = "WAE"
TIME_STEPPING_MODEL_TYPE = "FNO"
LOAD_MODEL_FROM_ORACLE = False

LATENT_DIM = 4

if PHASE == "single":
    NUM_STATES = 2
elif PHASE == "multi":
    NUM_STATES = 3


MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE"
ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_{LATENT_DIM}_consistency'

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

time_stepping_model_path = f"trained_models/time_steppers/{PHASE}_phase_{TIME_STEPPING_MODEL_TYPE}"
time_stepper = load_trained_time_stepping_model(
    model_load_path=time_stepping_model_path,
    device=DEVICE,
)
if TIME_STEPPING_MODEL_TYPE == 'FNO':
    input_seq_len = 32
else:
    input_seq_len = time_stepper.input_seq_len
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

SAMPLE_IDS = range(1, 2)

if LOCAL_OR_ORACLE == 'oracle':
    dataset = AEDataset(
        oracle_path=ORACLE_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        preprocessor=preprocessor,
        num_skip_steps=NUM_SKIP_STEPS,
    )
elif LOCAL_OR_ORACLE == 'local':
    dataset = AEDataset(
        local_path=LOCAL_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        preprocessor=preprocessor,
        num_skip_steps=NUM_SKIP_STEPS,
    )

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

def main():

    num_steps = 500

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
        plt.subplot(1, 3, 1)
        plt.plot(latent_state[0, 0, :num_steps], label='latent state', color='tab:blue')
        for i in range(1, num_latent_to_plot):
            plt.plot(latent_state[0, i, :num_steps], color='tab:blue')

        plt.plot(pred_latent_state[0, 0, :num_steps], label='pred latent state', color='tab:orange')
        for i in range(1, num_latent_to_plot):
            plt.plot(pred_latent_state[0, i, :num_steps], color='tab:orange') 
        plt.legend()

    time_step_to_plot_1 = 100
    time_step_to_plot_2 = 200
    plt.subplot(1, 3, 2)
    plt.plot(state[0, 0, :, time_step_to_plot_1], label='state', color='tab:blue')
    plt.plot(pred_recon_state[0, 0, :, time_step_to_plot_1], label='pred state', color='tab:orange')
    plt.plot(state[0, 0, :, time_step_to_plot_2], color='tab:blue')
    plt.plot(pred_recon_state[0, 0, :, time_step_to_plot_2], color='tab:orange')

    plt.subplot(1, 3, 3)
    plt.plot(state[0, 1, :, time_step_to_plot_1], label='state', color='tab:blue')
    plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_1], label='pred state', color='tab:orange')
    plt.plot(state[0, 1, :, time_step_to_plot_2], color='tab:blue')
    plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_2], color='tab:orange')
    plt.show()



if __name__ == "__main__":
    
    main()

import pdb
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from latent_time_stepping.datasets.AE_dataset import AEDataset

from latent_time_stepping.utils import load_trained_AE_model, load_trained_time_stepping_model
torch.set_default_dtype(torch.float32)


NUM_SKIP_STEPS = 5

DEVICE = 'cpu'

PHASE = "single"
AE_MODEL_TYPE = "WAE"
TIME_STEPPING_MODEL_TYPE = "transformer"

if PHASE == "single":
    NUM_STATES = 2
elif PHASE == "multi":
    NUM_STATES = 3

AE_model_path = f"trained_models/autoencoders/{PHASE}_phase_{AE_MODEL_TYPE}"
AE = load_trained_AE_model(
    model_load_path=AE_model_path,
    model_type=AE_MODEL_TYPE,
    device=DEVICE,
)

time_stepping_model_path = f"trained_models/time_steppers/{PHASE}_phase_{TIME_STEPPING_MODEL_TYPE}"
time_stepper = load_trained_time_stepping_model(
    model_load_path=time_stepping_model_path,
    device=DEVICE,
)
input_seq_len = time_stepper.input_seq_len

PREPROCESSOR_PATH = f'trained_preprocessors/{PHASE}_phase_preprocessor.pkl'
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

LOCAL_OR_ORACLE = 'oracle'
LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/training_data'

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/test'

SAMPLE_IDS = range(10, 11)

if LOCAL_OR_ORACLE == 'oracle':
    dataset = AEDataset(
        oracle_path=ORACLE_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        preprocessor=preprocessor,
    )
elif LOCAL_OR_ORACLE == 'local':
    dataset = AEDataset(
        local_path=LOCAL_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        preprocessor=preprocessor,
    )

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

def main():

    num_steps = 2000

    for i, (state, pars) in enumerate(dataloader):
        state = state.to(DEVICE)
        pars = pars.to(DEVICE)

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
    
    latent_state = latent_state.detach().numpy()
    pred_latent_state = pred_latent_state.detach().numpy()
    state = state.detach().numpy()
    pred_recon_state = pred_recon_state.detach().numpy()

    num_latent_to_plot = 2
    plt.figure()
    plt.plot(latent_state[0, 0, :num_steps], label='latent state', color='tab:blue')
    for i in range(1, num_latent_to_plot):
        plt.plot(latent_state[0, i, :num_steps], color='tab:blue')

    plt.plot(pred_latent_state[0, 0, :num_steps], label='pred latent state', color='tab:orange')
    for i in range(1, num_latent_to_plot):
        plt.plot(pred_latent_state[0, i, :num_steps], color='tab:orange')                        
    plt.legend()
    plt.show()



if __name__ == "__main__":
    
    main()

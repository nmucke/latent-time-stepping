import pdb
import pickle
from matplotlib.animation import FuncAnimation
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

ANIMATE = True

DEVICE = 'cpu'

PHASE = "wave"
AE_MODEL_TYPE = "WAE"
TIME_STEPPING_MODEL_TYPE = "transformer"
LOAD_MODEL_FROM_ORACLE = True

NUM_SKIP_STEPS = 4 if PHASE == 'single' else 10

if PHASE == 'single':
    num_skip_steps = 4
    LATENT_DIM = 4
    NUM_STATES = 2
    LOCAL_OR_ORACLE = 'oracle'
    LOAD_MODEL_FROM_ORACLE = False
elif PHASE == 'multi':
    num_skip_steps = 10
    LATENT_DIM = 8
    NUM_STATES = 3
    LOCAL_OR_ORACLE = 'oracle'
    LOAD_MODEL_FROM_ORACLE = True
elif PHASE == 'lorenz':
    num_skip_steps = 5
    LATENT_DIM = 16
    NUM_STATES = 1
    LOCAL_OR_ORACLE = 'local'
    LOAD_MODEL_FROM_ORACLE = True
elif PHASE == 'wave':
    num_skip_steps = 1
    LATENT_DIM = 8
    NUM_STATES = 2
    LOCAL_OR_ORACLE = 'local'
    LOAD_MODEL_FROM_ORACLE = False

#LATENT_DIM = 8 if PHASE == 'multi' else 4


if PHASE == 'multi':
    ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_8_latent_0.0001_consistency_0.01_channels_128_layers_6_trans_layers_2_embedding_64_vit' #'multi_phase/autoencoders/WAE_8_latent_0.001_consistency_0.01_channels_128_layers_6_trans_layers_1_embedding_64_vit'
elif PHASE == 'lorenz':
    #MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}"
    ORACLE_MODEL_LOAD_PATH = f'{PHASE}_phase/autoencoders/WAE_16_latent_0.001_consistency_0.01_channels_64_layers_3_trans_layers_1_embedding_64_vit'
elif PHASE == 'single':
    MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE_vit_conv_{LATENT_DIM}_1_trans_layer"
    ORACLE_MODEL_LOAD_PATH = None
elif PHASE == 'wave':
    MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_WAE_2_layers"
    ORACLE_MODEL_LOAD_PATH = None

object_storage_client = ObjectStorageClientWrapper(
    bucket_name='trained_models'
)

if ORACLE_MODEL_LOAD_PATH is not None:
    state_dict, config = object_storage_client.get_model(
        source_path=ORACLE_MODEL_LOAD_PATH,
        device=DEVICE,
    )
#model.load_state_dict(state_dict['model_state_dict'])
AE = load_trained_AE_model(
    model_load_path=MODEL_LOAD_PATH if not LOAD_MODEL_FROM_ORACLE else None,
    state_dict=state_dict if LOAD_MODEL_FROM_ORACLE else None,
    config=config if LOAD_MODEL_FROM_ORACLE else None,
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
    input_seq_len = 16#time_stepper.input_seq_len
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

LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/train'

BUCKET_NAME = "bucket-20230222-1753"
ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/test'

SAMPLE_IDS = range(0, 1)



dataset = AEDataset(
    oracle_path=ORACLE_LOAD_PATH if LOCAL_OR_ORACLE == 'oracle' else None,                                                          
    local_path=LOCAL_LOAD_PATH if LOCAL_OR_ORACLE == 'local' else None,
    sample_ids=SAMPLE_IDS,
    preprocessor=preprocessor,
    num_skip_steps=num_skip_steps,
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

    num_steps = 1200

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

        print(f'{i}: {preprocessor.inverse_transform_pars(pars, ensemble=True)}')
    
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
    #plt.plot(state[0, 0, :, time_step_to_plot_2], color='tab:blue', label=f'HF, t={time_step_to_plot_2}')
    #plt.plot(pred_recon_state[0, 0, :, time_step_to_plot_2], color='tab:orange')
    plt.plot(state[0, 0, :, time_step_to_plot_3], color='tab:blue', label=f'HF, t={time_step_to_plot_3}')
    plt.plot(pred_recon_state[0, 0, :, time_step_to_plot_3], color='tab:orange')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(state[0, 1, :, time_step_to_plot_1], color='tab:blue', label=f'HF, t={time_step_to_plot_1}')
    plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_1], label='pred state', color='tab:orange')
    #plt.plot(state[0, 1, :, time_step_to_plot_2], color='tab:blue', label=f'HF, t={time_step_to_plot_2}')
    #plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_2], color='tab:orange')
    plt.plot(state[0, 1, :, time_step_to_plot_3], color='tab:blue', label=f'HF, t={time_step_to_plot_3}')
    plt.plot(pred_recon_state[0, 1, :, time_step_to_plot_3], color='tab:orange')
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


    if ANIMATE:
        t_vec = np.arange(0, 45, 0.03535)
        t_vec = t_vec[:pred_recon_state.shape[-1]]

        x = np.linspace(0, 25.6, 512)

        fig, ax = plt.subplots()
        #xdata, ydata, ydata_1 = [], [], []
        ln, = ax.plot([], [], lw=3, animated=True)
        ln_1, = ax.plot([], [], '--', lw=3, animated=True)
        ax.grid()

        def init():
            ax.set_xlim(0, 25.6)
            ax.set_ylim(-0.025, 0.025)
            return ln,
    
        def update(frame):
            #xdata.append(x)
            #ydata.append(state[0, :, frame])
            #ydata_1.append(pred_recon_state[0, :, frame])
            ln.set_data(x, state[0, 0, :, frame], )
            ln_1.set_data(x, pred_recon_state[0, 0, :, frame],)
            plt.legend([f'High-fidelity', f'Neural network'])
            plt.xlabel('x')
            plt.ylabel('\eta')
            plt.title(f't = {t_vec[frame]:.2f}')
            return ln, ln_1,

        ani = FuncAnimation(
            fig,
            update,
            frames=len(t_vec),
            init_func=init, 
            blit=True,
            interval=10,
            )
        ani.save('submerged_bar.gif', fps=30)
        plt.show()

if __name__ == "__main__":
    
    main()

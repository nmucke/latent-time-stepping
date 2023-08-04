
import pdb
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import yaml
from latent_time_stepping.datasets.AE_dataset import AEDataset
from latent_time_stepping.preprocessor import Preprocessor
from latent_time_stepping.utils import load_trained_AE_model

torch.set_default_dtype(torch.float32)

DEVICE = 'cpu'

PHASE = "multi"
MODEL_TYPE = "WAE"

LOCAL_OR_ORACLE = 'oracle'

if PHASE == "single":
    NUM_STATES = 2
elif PHASE == "multi":
    NUM_STATES = 3

MODEL_LOAD_PATH = f"trained_models/autoencoders/{PHASE}_phase_{MODEL_TYPE}"
model = load_trained_AE_model(
    model_load_path=MODEL_LOAD_PATH,
    model_type=MODEL_TYPE,
    device=DEVICE,
)

PREPROCESSOR_PATH = f'trained_preprocessors/{PHASE}_phase_preprocessor.pkl'
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

LOCAL_LOAD_PATH = f'data/{PHASE}_phase/raw_data/training_data'
ORACLE_LOAD_PATH = f'{PHASE}_phase/raw_data/test'

NUM_SAMPLES = 2
SAMPLE_IDS = range(NUM_SAMPLES)

if LOCAL_OR_ORACLE == 'oracle':
    dataset = AEDataset(
        oracle_path=ORACLE_LOAD_PATH,
        sample_ids=SAMPLE_IDS,
        num_skip_steps=1,
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
    num_workers=NUM_SAMPLES,
)

def main():


    L2_error = []

    pbar = tqdm(
            enumerate(dataloader),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
    for i, (state, pars) in pbar:
        
        state = state.to(DEVICE)
        pars = pars.to(DEVICE)

        latent_state = model.encode(state)

        recon_state = model.decode(latent_state, pars)
        recon_state = preprocessor.inverse_transform_state(recon_state, ensemble=True)
        state = preprocessor.inverse_transform_state(state, ensemble=True)
        recon_state = recon_state.detach()
        
        L2_error.append(torch.norm(state - recon_state)/torch.norm(state))

    print(f"Average L2 Error: {torch.mean(torch.stack(L2_error))}")

    recon_state = recon_state[0].detach().numpy()
    hf_trajectory = state[0].detach().numpy()
    latent_state = latent_state.detach().numpy()
    
    normal_distribution = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.linspace(-5, 5, 1000) ** 2)
    
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 4, 1)
    plt.plot(recon_state[2, :, 100], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[2, :, 100], label="High Fidelity", color='tab:blue')
    plt.plot(recon_state[2, :, -1], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[2, :, -1], label="High Fidelity", color='tab:blue')
    plt.legend()
    plt.subplot(2, 4, 2)
    plt.plot(recon_state[1, :, 100], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[1, :, 100], label="High Fidelity", color='tab:blue')
    plt.plot(recon_state[1, :, -1], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[1, :, -1], label="High Fidelity", color='tab:blue')
    plt.legend()
    plt.subplot(2, 4, 3)
    plt.plot(latent_state[0, 0, :])
    plt.plot(latent_state[0, 1, :])
    plt.plot(latent_state[0, 2, :])
    plt.grid()
    plt.subplot(2, 4, 4)
    plt.plot(latent_state[0, 3, :])
    plt.plot(latent_state[0, 4, :])
    plt.plot(latent_state[0, 5, :])
    plt.grid()
    plt.subplot(2, 4, 5)
    plt.hist(latent_state[:, 0, :].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')
    plt.subplot(2, 4, 6)
    plt.hist(latent_state[:, 1, :].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')
    plt.subplot(2, 4, 7)
    plt.hist(latent_state[:, 2, :].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')
    plt.subplot(2, 4, 8)
    plt.hist(latent_state[:, 3, :].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')

    plt.show()


if __name__ == "__main__":
    
    main()


'''
if i == 5:

tt = 5
recon_state = model.decoder(latent_state[i], pars).detach().numpy()
WAE_latent_state = latent_state[i]
AE_latent_state = model_AE.encoder(hf_trajectory)

WAE_latent_state = WAE_latent_state[tt].unsqueeze(0)
AE_latent_state = AE_latent_state[tt].unsqueeze(0)
noise_std = 0.75
noise = torch.randn((500, 16))*noise_std

WAE_latent_state = WAE_latent_state + noise
AE_latent_state = AE_latent_state + noise

WAE_recon_state = model.decoder(WAE_latent_state, pars[0:500]).detach().numpy()
AE_recon_state = model_AE.decoder(AE_latent_state, pars[0:500]).detach().numpy()

WAE_recon_state_mean = WAE_recon_state.mean(axis=0)
AE_recon_state_mean = AE_recon_state.mean(axis=0)

WAE_recon_state_std = WAE_recon_state.std(axis=0)
AE_recon_state_std = AE_recon_state.std(axis=0)

plt.figure()
#plt.plot(recon_state[tt, 1, :], label="Reconstructed", color='tab:orange')
plt.plot(hf_trajectory[tt, 1, :], label="High-fidelity", color='tab:blue', lw=2)
plt.plot(WAE_recon_state_mean[1, :], label="WAE", color='tab:green', lw=2)
plt.fill_between(
np.arange(0, 256), 
WAE_recon_state_mean[1, :] - WAE_recon_state_std[1, :],
WAE_recon_state_mean[1, :] + WAE_recon_state_std[1, :],
color='tab:green',
alpha=0.2
)
plt.plot(AE_recon_state_mean[1, :], label="AE", color='tab:red', lw=2)
plt.fill_between(
np.arange(0, 256), 
AE_recon_state_mean[1, :] - AE_recon_state_std[1, :],
AE_recon_state_mean[1, :] + AE_recon_state_std[1, :],
color='tab:red',
alpha=0.2
)
plt.legend()
plt.grid()
plt.xlabel("Space")
plt.ylabel("Velocity")
plt.savefig(f"AE_vs_WAE_noise{noise_std}.pdf")
plt.show()
pdb.set_trace()

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(recon_state[tt, 0, :], label="Reconstructed", color='tab:orange')
plt.plot(hf_trajectory[tt, 0, :], label="High Fidelity", color='tab:blue')
plt.plot(WAE_recon_state_mean[0, :], label="WAE Reconstructed", color='tab:green')
plt.fill_between(
np.arange(0, 256), 
WAE_recon_state_mean[0, :] - WAE_recon_state_std[0, :],
WAE_recon_state_mean[0, :] + WAE_recon_state_std[0, :],
color='tab:green',
alpha=0.2
)
plt.plot(AE_recon_state_mean[0, :], label="AE Reconstructed", color='tab:red')
plt.fill_between(
np.arange(0, 256), 
AE_recon_state_mean[0, :] - AE_recon_state_std[0, :],
AE_recon_state_mean[0, :] + AE_recon_state_std[0, :],
color='tab:red',
alpha=0.2
)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(recon_state[tt, 1, :], label="Reconstructed", color='tab:orange')
plt.plot(hf_trajectory[tt, 1, :], label="High Fidelity", color='tab:blue')
plt.plot(WAE_recon_state_mean[1, :], label="WAE Reconstructed", color='tab:green')
plt.fill_between(
np.arange(0, 256), 
WAE_recon_state_mean[1, :] - WAE_recon_state_std[1, :],
WAE_recon_state_mean[1, :] + WAE_recon_state_std[1, :],
color='tab:green',
alpha=0.2
)
plt.plot(AE_recon_state_mean[1, :], label="AE Reconstructed", color='tab:red')
plt.fill_between(
np.arange(0, 256), 
AE_recon_state_mean[1, :] - AE_recon_state_std[1, :],
AE_recon_state_mean[1, :] + AE_recon_state_std[1, :],
color='tab:red',
alpha=0.2
)
plt.legend()
plt.show()
pdb.set_trace()
'''

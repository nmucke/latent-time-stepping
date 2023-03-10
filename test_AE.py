import pdb
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)

MODEL_TYPE = "WAE"
model = torch.load(f"trained_models/autoencoders/{MODEL_TYPE}.pt")
model = model.to('cpu')
model.eval()

STATE_PATH = 'data/raw_data/training_data/state/sample_'
PARS_PATH = 'data/raw_data/training_data/pars/sample_'

PREPROCESSOR_PATH = 'data/processed_data/trained_preprocessor.pt'

def main():
    num_samples = 10
    num_time_steps = 501#2001

    preprocessor = torch.load(PREPROCESSOR_PATH)

    latent_state = torch.zeros((
        num_samples, 
        num_time_steps,
        model.encoder.latent_dim,
        ))
    
    L2_error = []
    pbar = tqdm(range(num_samples), desc="Encoding Trajectories")

    pbar = tqdm(
            range(num_samples),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
    for i in pbar:
        state = np.load(f'{STATE_PATH}{i}.npy')
        hf_trajectory = torch.tensor(state, dtype=torch.get_default_dtype())
        hf_trajectory = hf_trajectory[:, :, 0::4]
        hf_trajectory = preprocessor.transform_state(hf_trajectory)
        hf_trajectory = hf_trajectory.transpose(1, 2)
        hf_trajectory = hf_trajectory.transpose(0, 1)

        pars = np.load(f'{PARS_PATH}{i}.npy')
        pars = torch.tensor(pars, dtype=torch.get_default_dtype())
        pars = preprocessor.transform_pars(pars)
        pars = pars.repeat(num_time_steps, 1)
        
        if MODEL_TYPE == "WAE":
            latent_state[i] = model.encode(hf_trajectory)
        elif MODEL_TYPE == "VAE":
            latent_state[i], _, _ = model.encoder(hf_trajectory)

        recon_state = model.decoder(latent_state[i], pars)
        
        L2_error.append(torch.norm(hf_trajectory - recon_state)/torch.norm(hf_trajectory))

    print(f"Average L2 Error: {torch.mean(torch.stack(L2_error))}")

    recon_state = recon_state.detach().numpy()
    hf_trajectory = hf_trajectory.detach().numpy()
    latent_state = latent_state.detach().numpy()
    
    normal_distribution = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.linspace(-5, 5, 1000) ** 2)
    
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 4, 1)
    plt.plot(recon_state[100, 0, :], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[100, 0, :], label="High Fidelity", color='tab:blue')
    plt.plot(recon_state[-1, 0, :], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[-1, 0, :], label="High Fidelity", color='tab:blue')
    plt.legend()
    plt.subplot(2, 4, 2)
    plt.plot(recon_state[100, 1, :], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[100, 1, :], label="High Fidelity", color='tab:blue')
    plt.plot(recon_state[-1, 1, :], label="Reconstructed", color='tab:orange')
    plt.plot(hf_trajectory[-1, 1, :], label="High Fidelity", color='tab:blue')
    plt.legend()
    plt.subplot(2, 4, 3)
    plt.plot(latent_state[0, :, 0])
    plt.plot(latent_state[0, :, 1])
    plt.plot(latent_state[0, :, 2])
    plt.grid()
    plt.subplot(2, 4, 4)
    plt.plot(latent_state[0, :, 3])
    plt.plot(latent_state[0, :, 4])
    plt.plot(latent_state[0, :, 5])
    plt.grid()
    plt.subplot(2, 4, 5)
    plt.hist(latent_state[:, :, 0].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')
    plt.subplot(2, 4, 6)
    plt.hist(latent_state[:, :, 1].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')
    plt.subplot(2, 4, 7)
    plt.hist(latent_state[:, :, 2].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')
    plt.subplot(2, 4, 8)
    plt.hist(latent_state[:, :, 3].flatten(), bins=50, density=True)
    plt.plot(np.linspace(-5, 5, 1000), normal_distribution,color='tab:red')

    plt.show()


if __name__ == "__main__":
    
    main()

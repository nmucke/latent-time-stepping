import pdb
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)

MODEL_TYPE = "WAE"
model = torch.load(f"trained_models/autoencoders/{MODEL_TYPE}.pt")
model = model.to('cpu')

STATE_PATH = 'data/processed_data/training_data/states.pt'

PREPROCESSOR_PATH = 'data/processed_data/trained_preprocessor.pt'

def main():
    state = torch.load(STATE_PATH)

    latent_state = torch.zeros((
        state.shape[0], 
        model.encoder.latent_dim,
        state.shape[-1]
        ))
    
    L2_error = []
    pbar = tqdm(range(state.shape[0]), desc="Encoding Trajectories")
    for i in pbar:
        hf_trajectory = state[i]
        hf_trajectory = hf_trajectory.transpose(1, 2)
        hf_trajectory = hf_trajectory.transpose(0, 1)
    
        latent_state[i] = model.encode(hf_trajectory)

        recon_state = model.decode(latent_state[i])

        recon_state = recon_state.transpose(0, 1)
        recon_state = recon_state.transpose(1, 2)

        L2_error.append(torch.norm(hf_trajectory - recon_state))

    print(f"Average L2 Error: {torch.mean(torch.stack(L2_error))}")

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.plot(recon_state[0, 0, :])
    plt.plot(hf_trajectory[0, 0, :])
    plt.subplot(1, 4, 2)
    plt.plot(recon_state[0, 1, :])
    plt.plot(hf_trajectory[0, 1, :])
    plt.subplot(1, 4, 3)
    plt.hist(latent_state[:, 0])
    plt.subplot(1, 4, 4)
    plt.hist(latent_state[:, 1])
    plt.show()


if __name__ == "__main__":
    
    main()

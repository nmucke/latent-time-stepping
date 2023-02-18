import pdb
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)

MODEL_TYPE = "WAE"
model = torch.load(f"trained_models/autoencoders/{MODEL_TYPE}.pt")
model = model.to('cuda')

STATE_PATH = 'data/processed_data/training_data/states.pt'


def main():
    state = torch.load(STATE_PATH)

    latent_state = torch.zeros((
        state.shape[0], 
        model.encoder.latent_dim,
        state.shape[-1]
        ))
    
    pbar = tqdm(range(state.shape[0]), desc="Encoding Trajectories")
    for i in pbar:
        hf_trajectory = state[i]
        hf_trajectory = hf_trajectory.transpose(1, 2)
        hf_trajectory = hf_trajectory.transpose(0, 1)

        hf_trajectory = hf_trajectory.to('cuda')
    
        latent_state[i] = model.encode(hf_trajectory).transpose(0, 1).to('cpu').detach()

    torch.save(latent_state, f"data/processed_data/training_data/latent_states.pt")

if __name__ == "__main__":
    
    main()

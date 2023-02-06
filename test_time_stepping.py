import pdb
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)

AE_TYPE = "WAE"
AE = torch.load(f"trained_models/autoencoders/{AE_TYPE}.pt")
AE = AE.to('cpu')
AE.eval()

time_stepper = torch.load(f"trained_models/time_steppers/time_stepping.pt")
time_stepper = time_stepper.to('cpu')
time_stepper.eval()

STATE_PATH = 'data/raw_data/test_data/state/sample_'
PARS_PATH = 'data/raw_data/test_data/pars/sample_'

PREPROCESSOR_PATH = 'data/processed_data/trained_preprocessor.pt'

def main():
    case = 2

    num_time_steps = 2001

    time = np.linspace(0, 100, num_time_steps)
    time = time[0::4]

    x_vec = np.linspace(0, 1000, 256)

    preprocessor = torch.load(PREPROCESSOR_PATH)

    state = np.load(f'{STATE_PATH}{case}.npy')
    state = state[:, :, 0::4]
    pars = np.load(f'{PARS_PATH}{case}.npy')
    pars = preprocessor.transform_pars(pars)
    pars = torch.tensor(pars, dtype=torch.get_default_dtype())
    pars = pars.unsqueeze(0)
    
    hf_trajectory = torch.tensor(state, dtype=torch.get_default_dtype())
    hf_trajectory = preprocessor.transform_state(hf_trajectory)
    hf_trajectory = hf_trajectory.transpose(1, 2)
    hf_trajectory = hf_trajectory.transpose(0, 1)
    
    latent_state = AE.encode(hf_trajectory)
    latent_state = latent_state.unsqueeze(0)

    init_time_steps = 50
    output_seq_len = 100
    pred_latent_state = time_stepper.multistep_prediction(
        latent_state[:, 0:init_time_steps],
        pars,
        output_seq_len=output_seq_len,
        )
    pred_hf_trajectory = AE.decoder(pred_latent_state.squeeze(0), pars.repeat(init_time_steps+output_seq_len, 1))

    time_plotting = time[0:(init_time_steps + output_seq_len)]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        time_plotting, pred_latent_state[0, :, 0].detach().numpy(), 
        label="Predicted", color='tab:orange', linewidth=2
        )
    plt.plot(
        time_plotting, latent_state[0, 0:len(time_plotting), 0].detach().numpy(), 
        label="Actual", color='tab:blue', linewidth=2
        )

    plt.plot(
        time_plotting, pred_latent_state[0, :, 1].detach().numpy(), 
        color='tab:orange', linewidth=2
        )
    plt.plot(
        time_plotting, latent_state[0, 0:len(time_plotting), 1].detach().numpy(), 
        color='tab:blue', linewidth=2
        )
    plt.legend()
    plt.grid()


    plt.subplot(1, 2, 2)

    t1, t2, t3 = 10, len(time_plotting)//2, len(time_plotting)-1
    plt.plot(
        x_vec, pred_hf_trajectory[t1, 1, :].detach().numpy(), 
        label="Predicted", color='tab:orange', linewidth=2
        )
    plt.plot(
        x_vec, hf_trajectory[t1, 1, :].detach().numpy(), 
        label="Actual", color='tab:blue', linewidth=2
        )
    
    plt.plot(
        x_vec, pred_hf_trajectory[t2, 1, :].detach().numpy(), 
        color='tab:orange', linewidth=2
        )
    plt.plot(
        x_vec, hf_trajectory[t2, 1, :].detach().numpy(), 
        color='tab:blue', linewidth=2
        )
    
    plt.plot(
        x_vec, pred_hf_trajectory[t3, 1, :].detach().numpy(), 
        color='tab:orange', linewidth=2
        )
    plt.plot(
        x_vec, hf_trajectory[t3, 1, :].detach().numpy(), 
        color='tab:blue', linewidth=2
        )
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    main()

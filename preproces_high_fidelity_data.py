import torch
from tqdm import tqdm
import numpy as np
import pdb
import pickle

import matplotlib.pyplot as plt

from latent_time_stepping.preprocessor import Preprocessor

torch.set_default_dtype(torch.float32)

DATA_PATH = 'data/raw_data/training_data'
state_path = f'{DATA_PATH}/state/sample_'
pars_path = f'{DATA_PATH}/pars/sample_'

TRAINED_PREPROCESSOR_SAVE_PATH = 'data/processed_data/trained_preprocessor.pt'

def main():

    preprocessor = Preprocessor(num_states=2, num_pars=2)

    sample_ids = range(3000)

    # Fit the preprocessor
    pbar = tqdm(sample_ids, total=len(sample_ids))
    for sample_id in pbar:
        state = np.load(f'{state_path}{sample_id}.npy')
        pars = np.load(f'{pars_path}{sample_id}.npy')
        
        state = torch.from_numpy(state)
        pars = torch.from_numpy(pars)

        state = state.type(torch.get_default_dtype())
        pars = pars.type(torch.get_default_dtype())

        preprocessor.partial_fit_state(state)
        preprocessor.partial_fit_pars(pars)
    
    # Save the preprocessor   
    torch.save(preprocessor, TRAINED_PREPROCESSOR_SAVE_PATH) 

    # Transform the data
    processed_states = []
    processed_pars = []
    pbar = tqdm(sample_ids, total=len(sample_ids))
    for sample_id in pbar:
        state = np.load(f'{state_path}{sample_id}.npy')
        pars = np.load(f'{pars_path}{sample_id}.npy')

        state = torch.from_numpy(state).type(torch.get_default_dtype())
        pars = torch.from_numpy(pars).type(torch.get_default_dtype())

        preprocessor.transform_state(state)
        preprocessor.transform_pars(pars)

        processed_states.append(state)
        processed_pars.append(pars)
    
    processed_states = torch.stack(processed_states)
    processed_pars = torch.stack(processed_pars)

    # Save the processed data
    torch.save(processed_states, 'data/processed_data/training_data/states.pt')
    torch.save(processed_pars, 'data/processed_data/training_data/pars.pt')

if __name__ == "__main__":
    
    main()

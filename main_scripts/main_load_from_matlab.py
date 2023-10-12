from scipy.io import loadmat
import os
import pdb
import numpy as np

from latent_time_stepping.utils import create_directory


def main():

    num_samples = 150
    train_or_test = 'train'
    
    data_load_path = f'../allan/2DBarTestExperiment/data/{train_or_test}'
    data_save_path = f'data/wave_phase/raw_data/{train_or_test}'

    create_directory(f'{data_save_path}/state')
    create_directory(f'{data_save_path}/pars')

    for idx in range(num_samples):
        state = loadmat(f'{data_load_path}/state/sample_{idx}.mat')['state']
        pars = loadmat(f'{data_load_path}/pars/sample_{idx}.mat')['pars'][0]

        np.savez_compressed(f'{data_save_path}/state/sample_{idx}', data=state)
        np.savez_compressed(f'{data_save_path}/pars/sample_{idx}', data=pars)

if __name__ == '__main__':
    main()
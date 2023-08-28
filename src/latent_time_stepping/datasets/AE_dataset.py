import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt

from latent_time_stepping.oracle import ObjectStorageClientWrapper
from latent_time_stepping.preprocessor import Preprocessor
from scipy.signal import savgol_filter

class AEDataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(
        self,
        local_path: str = None,
        oracle_path: str = None,
        preprocessor: Preprocessor = None,
        load_entire_dataset: bool = False,
        num_skip_steps: int = 1,
        end_time_index: float = None,
        sample_ids: int = None,
        save_to_local: str = None,
        save_to_oracle: str = None,
        num_random_idx_divisor: int = None,
        states_to_include: tuple = None,
        filter: bool = False,
        ) -> None:
        super().__init__()

        self.sample_ids = sample_ids
        self.num_skip_steps = num_skip_steps
        self.end_time_index = end_time_index
        self.num_random_idx_divisor = num_random_idx_divisor
        self.states_to_include = states_to_include

        self.filter = filter

        self.save_to_local = save_to_local
        self.save_to_oracle = save_to_oracle

        self.preprocessor = preprocessor
        self.load_entire_dataset = load_entire_dataset

        self.oracle_path = oracle_path
        self.local_path = local_path

        if oracle_path is not None:
            bucket_name = "bucket-20230222-1753"

            self.oracle_path = oracle_path
            self.object_storage_client = ObjectStorageClientWrapper(bucket_name)

        elif local_path is not None:
            self.local_path = local_path

        if self.load_entire_dataset:
            self._load_entire_dataset()
    

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _get_oracle_data_sample(self, index: int):
        
        state = self.object_storage_client.get_numpy_object(
            source_path=f'{self.oracle_path}/state/sample_{index}.npz'
        )
        
        pars = self.object_storage_client.get_numpy_object(
            source_path=f'{self.oracle_path}/pars/sample_{index}.npz'
        )

        if self.end_time_index is not None:
            state = state[:, :, :self.end_time_index]

        state = state[:, :, 0::self.num_skip_steps]

        #state = state.astype('float32')
        #pars = pars.astype('float32')                                                   
        
        state = torch.tensor(state, dtype=torch.get_default_dtype())
        pars = torch.tensor(pars, dtype=torch.get_default_dtype())

        return state, pars
    
    def _get_local_data_sample(self, index: int):                                                      
        
        state = np.load(f'{self.local_path}/state/sample_{index}.npz')['data']
        pars = np.load(f'{self.local_path}/pars/sample_{index}.npz')['data']
        if self.end_time_index is not None:
            state = state[:, :, :self.end_time_index]

        state = state[:, :, 0::self.num_skip_steps]

        state = torch.tensor(state, dtype=torch.get_default_dtype())
        pars = torch.tensor(pars, dtype=torch.get_default_dtype())

        return state, pars
    
    def _load_entire_dataset(self,):

        if self.oracle_path is not None:


            self.state = self.object_storage_client.get_numpy_object(
                source_path=f'{self.oracle_path}/states.npz'
            )[self.sample_ids]
            
            self.pars = self.object_storage_client.get_numpy_object(
                source_path=f'{self.oracle_path}/pars.npz'
            )[self.sample_ids]

        elif self.local_path is not None:
            self.state = np.load(
                f'{self.local_path}/states.npz'
            )['data'][self.sample_ids]
            
            self.pars = np.load(
                f'{self.local_path}/pars.npz'
            )['data'][self.sample_ids]
            
        print(f"Loaded entire dataset. Shape: {self.state.shape}")
        self.state = torch.tensor(self.state, dtype=torch.get_default_dtype())
        self.pars = torch.tensor(self.pars, dtype=torch.get_default_dtype())   

    def __getitem__(self, index: int) -> torch.Tensor:    

        index = self.sample_ids[index]
            
        if self.load_entire_dataset:
            state = self.state[index]
            pars = self.pars[index]

        else:
            if self.oracle_path is not None:
                state, pars = self._get_oracle_data_sample(index)

            else:
                state, pars = self._get_local_data_sample(index)

        if self.preprocessor is not None:
            state = self.preprocessor.transform_state(state)
            pars = self.preprocessor.transform_pars(pars)
        
        
        if self.save_to_local is not None:
            
            state = state.numpy()
            pars = pars.numpy()

            np.savez_compressed(
                f'{self.save_to_local}/state/sample_{index}.npz',
                data=state
            )
            np.savez_compressed(
                f'{self.save_to_local}/pars/sample_{index}.npz',
                data=pars
            )

        if self.save_to_oracle is not None:

            state = state.numpy()
            pars = pars.numpy()

            self.object_storage_client.put_numpy_object(
                destination_path=f'{self.save_to_oracle}/state/sample_{index}.npz',
                data=state
            )
            self.object_storage_client.put_numpy_object(
                destination_path=f'{self.save_to_oracle}/pars/sample_{index}.npz',
                data=pars
            )

        if self.states_to_include is not None:
            state = state[self.states_to_include, :, :]

        if self.num_random_idx_divisor is not None:
            idx = np.random.choice(state.shape[-1], size=state.shape[-1]//self.num_random_idx_divisor, replace=False)

            state = state[:, :, idx]

        if self.filter:
            state = state.numpy()
            state = savgol_filter(state, 15, 1, axis=-1)
            state = torch.tensor(state)
        
        return state, pars

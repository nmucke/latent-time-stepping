import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt

from latent_time_stepping.oracle import ObjectStorageClientWrapper

  

class FNOTimeSteppingDataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(
        self,
        oracle_path: str = None,
        local_path: str = None,
        sample_ids: list = None,
        input_seq_len: int = 10,
        output_seq_len: int = 10,
        num_time_steps: int = 100,
        num_skip_steps: int = 1,
        preprocessor=None,
        ) -> None:
        super().__init__()

        self.oracle_path = None
        self.local_path = None
        self.sample_ids = sample_ids
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.num_time_steps = num_time_steps
        self.num_skip_steps = num_skip_steps

        self.preprocessor = preprocessor


        if oracle_path is not None:
            bucket_name = "bucket-20230222-1753"

            self.oracle_path = oracle_path
            self.object_storage_client = ObjectStorageClientWrapper(bucket_name)

        elif local_path is not None:
            self.local_path = local_path
        
    def _get_local_data_sample(self, index: int):                                                      
        
        state = np.load(f'{self.local_path}/state/sample_{index}.npz')['data']
        pars = np.load(f'{self.local_path}/pars/sample_{index}.npz')['data']

        state = state[:, :, 0::self.num_skip_steps]
        state = state[:, :, :self.num_time_steps]

        state = torch.tensor(state, dtype=torch.get_default_dtype())
        pars = torch.tensor(pars, dtype=torch.get_default_dtype())

        return state, pars
           
    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> torch.Tensor:

        state, pars = self._get_local_data_sample(index)

        if self.preprocessor is not None:
            state = self.preprocessor.transform_state(state)
            pars = self.preprocessor.transform_pars(pars)
        

        input_state = state[:, :, 0:-1]
        output_state = state[:, :, 1:]


        return input_state, output_state, pars
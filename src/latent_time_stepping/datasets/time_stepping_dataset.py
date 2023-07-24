import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt

from latent_time_stepping.oracle import ObjectStorageClientWrapper

class TimeSteppingDataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(
        self,
        oracle_path: str = None,
        local_path: str = None,
        sample_ids: list = None,
        input_seq_len: int = 10,
        output_seq_len: int = 10,
        ) -> None:
        super().__init__()

        self.oracle_path = None
        self.local_path = None
        self.sample_ids = sample_ids

        if oracle_path is not None:
            bucket_name = "bucket-20230222-1753"

            self.oracle_path = oracle_path
            self.object_storage_client = ObjectStorageClientWrapper(bucket_name)

        elif local_path is not None:
            self.local_path = local_path

        self._load_entire_dataset()

        self.input_state, self.output_state = self._prepare_multistep_state(
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len
            )
        
        
        self.pars = self.pars.unsqueeze(1)
        self.pars = self.pars.repeat(1, self.input_state.shape[1], 1)
        self.pars = self.pars.reshape(
            -1,
            self.pars.shape[-1],
            )

        self.input_state = self.input_state.reshape(
            self.input_state.shape[0] * self.input_state.shape[1],
            self.input_state.shape[2],
            self.input_state.shape[3],
            )
        self.output_state = self.output_state.reshape(
            self.output_state.shape[0] * self.output_state.shape[1],
            self.output_state.shape[2],
            self.output_state.shape[3],
            )
        
    def _load_entire_dataset(self,):
        
        if self.oracle_path is not None:
            self.state = self.object_storage_client.get_object(
                source_path=f'{self.oracle_path}/states.npz'
            )[self.sample_ids]
            
            self.pars = self.object_storage_client.get_object(
                source_path=f'{self.oracle_path}/pars.npz'
            )[self.sample_ids]

        elif self.local_path is not None:
            self.state = np.load(
                f'{self.local_path}/states.npz'
            )['data'][self.sample_ids]
            
            self.pars = np.load(
                f'{self.local_path}/pars.npz'
            )['data'][self.sample_ids]
            
        self.state = torch.tensor(self.state, dtype=torch.get_default_dtype())
        self.pars = torch.tensor(self.pars, dtype=torch.get_default_dtype())   
    
    def _prepare_multistep_state(
        self, 
        input_seq_len: int = 10,
        output_seq_len: int = 10,
        ):

        input_state = torch.zeros((
            self.state.shape[0],
            self.state.shape[-1] - input_seq_len - output_seq_len,
            self.state.shape[1],
            input_seq_len,
            ))
        output_state = torch.zeros((
            self.state.shape[0],
            self.state.shape[-1] - input_seq_len - output_seq_len,
            self.state.shape[1],
            output_seq_len,
            ))
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[-1] - input_seq_len - output_seq_len):
                input_state[i, j] = self.state[i, :, j:j+input_seq_len]
                output_state[i, j] = self.state[i, :, j+input_seq_len:j+input_seq_len+output_seq_len]

        return input_state, output_state

       
    def __len__(self) -> int:
        return self.pars.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:

        input_state = self.input_state[index]
        output_state = self.output_state[index]
        pars = self.pars[index]

        return input_state, output_state, pars
import torch
import pdb
import matplotlib.pyplot as plt

class TimeSteppingDataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        max_seq_len: int,
        num_skip_steps: int = 2,
        ) -> None:
        super().__init__()

        state = state.to(torch.get_default_dtype())
        pars = pars.to(torch.get_default_dtype())

        num_skip_steps = num_skip_steps

        state = state[:, :, 0::num_skip_steps]

        self.input_state, self.output_state = self._prepare_multistep_state(
            state=state,
            max_seq_len=max_seq_len,
            )
        
        self.pars = pars.unsqueeze(1)
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
        self.input_state = self.input_state.transpose(1, 2)
        self.output_state = self.output_state.transpose(1, 2)

    
    def _prepare_multistep_state(
        self, 
        state: torch.Tensor, 
        max_seq_len: int,
        ):

        input_state = torch.zeros((
            state.shape[0],
            state.shape[-1] - max_seq_len-1,
            state.shape[1],
            max_seq_len,
            ))
        output_state = torch.zeros((
            state.shape[0],
            state.shape[-1] - max_seq_len-1,
            state.shape[1],
            max_seq_len,
            ))
        for j in range(state.shape[0]):
            for i in range(state.shape[-1] - max_seq_len -1 ):
                input_state[j, i, :, :] = state[j, :, i:i+max_seq_len]
                output_state[j, i, :, :] = state[j, :, (i+1):(i+max_seq_len+1)]
        
        return input_state, output_state


        '''
        input_state = torch.zeros((
            state.shape[0],
            state.shape[-1] - input_seq_len - output_seq_len,
            state.shape[1],
            input_seq_len,
            ))
        output_state = torch.zeros((
            state.shape[0]*(state.shape[-1]-1),
            state.shape[-1] - input_seq_len - output_seq_len,
            state.shape[1],
            output_seq_len,
            ))
            
        return input_state, output_state
        '''

       
    def __len__(self) -> int:
        return self.pars.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:

        input_state = self.input_state[index]
        output_state = self.output_state[index]
        pars = self.pars[index]

        return input_state, output_state, pars

def get_time_stepping_dataloader(
    state: torch.Tensor,
    pars: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_skip_steps: int = 2,
    ) -> torch.utils.data.DataLoader:
    """Get the dataloader for the autoencoder."""

    dataset = TimeSteppingDataset(
        state=state,
        pars=pars,
        max_seq_len=max_seq_len,
        num_skip_steps=num_skip_steps,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        )

    return dataloader
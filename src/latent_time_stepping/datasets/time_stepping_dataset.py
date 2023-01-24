import torch
import pdb
import matplotlib.pyplot as plt

class TimeSteppingDataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        input_seq_len: int,
        output_seq_len: int,
        ) -> None:
        super().__init__()

        state = state.to(torch.get_default_dtype())
        pars = pars.to(torch.get_default_dtype())

        state = state[:, :, 0::4]

        self.input_state, self.output_state = self._prepare_multistep_state(
            state=state,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len
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
        input_seq_len: int, 
        output_seq_len: int
        ):

        
        
        input_state = torch.zeros((
            state.shape[0],
            state.shape[-1] - input_seq_len - output_seq_len,
            state.shape[1],
            input_seq_len,
            ))
        output_state = torch.zeros((
            state.shape[0],
            state.shape[-1] - input_seq_len - output_seq_len,
            state.shape[1],
            output_seq_len,
            ))
        for i in range(state.shape[0]):
            for j in range(state.shape[-1] - input_seq_len - output_seq_len):
                input_state[i, j] = state[i, :, j:j+input_seq_len]
                output_state[i, j] = state[i, :, j+input_seq_len:j+input_seq_len+output_seq_len]

        return input_state, output_state
       
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
    input_seq_len: int,
    output_seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    ) -> torch.utils.data.DataLoader:
    """Get the dataloader for the autoencoder."""

    dataset = TimeSteppingDataset(
        state=state,
        pars=pars,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        )

    return dataloader
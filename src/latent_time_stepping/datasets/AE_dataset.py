import torch
import pdb
import matplotlib.pyplot as plt

class AEDataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(
        self,
        state: torch.Tensor,
        pars: torch.Tensor,
        include_time: bool = False,
        num_skip_steps: int = 4,
        ) -> None:
        super().__init__()

        self.num_skip_steps = num_skip_steps

        self.state = state
        self.pars = pars
        self.include_time = include_time

        self.state = self.state.to(torch.get_default_dtype())
        self.pars = self.pars.to(torch.get_default_dtype())

        self._prepare_state_and_pars()


    def _prepare_state_and_pars(self,):
        self.state = self.state[:, : , :, 0::self.num_skip_steps]

        if self.include_time:
            self.time = torch.linspace(
                0,
                1000,
                self.state.shape[-1],
            )

            self.time = self.time.unsqueeze(0)
            self.time = self.time.repeat(self.state.shape[0], 1)
            self.time = self.time.flatten()

        self.state = self.state.transpose(2, 3)
        self.state = self.state.transpose(1, 2)

        self.pars = self.pars.unsqueeze(1)
        self.pars = self.pars.repeat(1, self.state.shape[1], 1)
        self.pars = self.pars.reshape(
            -1,
            self.pars.shape[-1],
            )

        self.state = self.state.reshape(
            -1,
            self.state.shape[2],
            self.state.shape[3],
            )

    def __len__(self) -> int:
        return self.pars.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:

        state = self.state[index]

        pars = self.pars[index]

        if self.include_time:
            time = self.time[index]
            return state, pars, time
            
        else:
            return state, pars, 0

def get_AE_dataloader(
    state: torch.Tensor,
    pars: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    include_time: bool,
    num_skip_steps: int = 4,
    ) -> torch.utils.data.DataLoader:
    """Get the dataloader for the autoencoder."""

    dataset = AEDataset(
        state=state,
        pars=pars,
        include_time=include_time,
        num_skip_steps=num_skip_steps,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        )

    return dataloader
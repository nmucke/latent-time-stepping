import pdb
from pickletools import optimize
import numpy as np
import torch
import torch.optim as optim


class Optimizer():

    def __init__(
        self,
        model: torch.nn.Module,
        args: dict,
    ) -> None:
    
        self.model = model
        self.args = args

        self.encoder = torch.optim.Adam(
            self.model.encoder.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        self.decoder = torch.optim.Adam(
            self.model.decoder.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        '''
        self.encoder_scheduler = CosineWarmupScheduler(
            self.encoder,
            **self.args['scheduler_args']
        )

        self.decoder_scheduler = CosineWarmupScheduler(
            self.decoder,
            **self.args['scheduler_args']
        )
        '''
        self.encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder,
            mode='min',
            factor=0.9,
            patience=5,
            verbose=True,
        )

        self.decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.decoder,
            mode='min',
            factor=0.9,
            patience=5,
            verbose=True,
        )

    def load_state_dict(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])

        self.encoder_scheduler.load_state_dict(state_dict['encoder_scheduler'])
        self.decoder_scheduler.load_state_dict(state_dict['decoder_scheduler'])
    
    def zero_grad(self) -> None:
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self) -> None:
        self.encoder.step()
        self.decoder.step()

    def step_scheduler(self, loss: float) -> None:
        self.encoder_scheduler.step(loss)
        self.decoder_scheduler.step(loss)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
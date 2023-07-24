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

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args['learning_rate'],
            weight_decay=self.args['weight_decay'],
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            **self.args['scheduler_args']
        )

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
    
    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def step_scheduler(self, loss: float) -> None:
        self.scheduler.step()

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
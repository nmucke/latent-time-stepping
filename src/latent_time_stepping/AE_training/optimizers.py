import torch


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

        self.encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
        )

        self.decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.decoder,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
        )
    
    def zero_grad(self) -> None:
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self) -> None:
        self.encoder.step()
        self.decoder.step()

    def step_scheduler(self, loss: float) -> None:
        self.encoder_scheduler.step(loss)
        self.decoder_scheduler.step(loss)
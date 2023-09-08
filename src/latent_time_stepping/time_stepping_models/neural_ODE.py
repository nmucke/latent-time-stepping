import math
import pdb
from typing import Any, Mapping
import torch
import torch.nn as nn
from latent_time_stepping.time_stepping_models.parameter_encoder import ParameterEncoder
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint


class NODE(nn.Module):
        def __init__(
            self,
            latent_dim: int,
            embed_dim: int,
            num_layers: int,
            pars: torch.Tensor = None,
            **kwargs: Any,
        ):
            super().__init__()
    
            self.latent_dim = latent_dim
            self.embed_dim = embed_dim
            self.num_layers = num_layers

            self.activation = nn.LeakyReLU()

            self.pars = pars

            self.network = nn.Sequential(
                nn.Linear(latent_dim + latent_dim, embed_dim),
                *[
                    nn.Sequential(
                        self.activation,
                        nn.Linear(
                            embed_dim, 
                            embed_dim if i < num_layers - 1 else latent_dim,
                            bias=True if i < num_layers - 1 else False,
                        ),
                    ) for i in range(num_layers)    
                ],
            )
    
        def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

            return self.network(torch.cat([z, self.pars], dim=-1))

class NODETimeSteppingModel(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_pars: int,
        embed_dim: int,
        num_layers: int,
        integrator: str,
        step_size: float,
        **kwargs: Any,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_pars = num_pars
        self.integrator = integrator
        self.step_size = step_size

        self.activation = nn.LeakyReLU()

        self.pars_encoder = nn.Sequential(
            nn.Linear(num_pars, latent_dim),
            self.activation,
            nn.Linear(latent_dim, latent_dim),
            self.activation,
        )

        self.network = NODE(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

    def encode_pars(self, pars: torch.Tensor) -> torch.Tensor:
        return self.pars_encoder(pars)
        

    def forward(
        self, 
        t: torch.Tensor,
        x: torch.Tensor,
        ) -> torch.Tensor:

        return self.network(torch.cat([x, self.pars], dim=-1))

    def multistep_prediction(
        self,
        input: torch.Tensor,
        pars: torch.Tensor,
        output_seq_len: int,
        ) -> torch.Tensor:

        input = input[:, :, -1]
        pars = self.encode_pars(pars)

        self.network.pars = pars
        
        out = odeint(
            func=self.network,
            y0=input,
            t=torch.arange(0, output_seq_len*self.step_size, self.step_size, device=self.device),
            method=self.integrator,
        )
        
        out = out.permute(1, 2, 0)

        return out
    
    
    @property
    def device(self):
        return next(self.parameters()).device




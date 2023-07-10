import pdb
import torch
import torch.nn as nn

from latent_time_stepping.time_stepping_models.transformer import (
    EncoderLayer,
    PositionalEmbedding
)

class ParameterEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        embed_hidden_dim: int,
        num_layers: int,
        pars_dim: int,
        p=0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embed_hidden_dim = embed_hidden_dim

        self.num_layers = num_layers

        self.pars_dim = pars_dim

        self.activation = nn.LeakyReLU()

        self.initial_dense_layer = nn.Linear(
            in_features=pars_dim,
            out_features=embed_dim
        )
        self.unflatten = nn.Unflatten(
            dim=1, 
            unflattened_size=(1, embed_dim)
            )

        self.out_dense_layer = nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim
        )
        
        '''
        self.initial_conv_layer = nn.Conv1d(
            in_channels=1,
            out_channels=input_seq_len,
            bias=True,
            kernel_size=1
        )

        self.positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=input_seq_len
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    embed_hidden_dim=embed_hidden_dim,
                    p=p
                )
                for _ in range(num_layers)
            ]
        )
        '''

    def forward(self, x):
        # x: (batch_size, pars_dim)
        x = self.initial_dense_layer(x)
        x = self.activation(x)
        x = self.out_dense_layer(x)
        x = self.unflatten(x) # (batch_size, 1, embed_dim)
        return x









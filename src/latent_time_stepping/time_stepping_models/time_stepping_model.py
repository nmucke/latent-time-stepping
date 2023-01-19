import pdb
import torch
import torch.nn as nn
from latent_time_stepping.time_stepping_models.parameter_encoder import ParameterEncoder

from latent_time_stepping.time_stepping_models.transformer import (
    EncoderLayer,
    DecoderLayer,
    PositionalEmbedding,
)

def create_look_ahead_mask(
    input_seq_len, 
    output_seq_len, 
    device='cpu'
    ):
    look_ahead_mask = torch.ones((output_seq_len, output_seq_len), device=device)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    past_mask = torch.zeros((output_seq_len, input_seq_len), device=device)

    look_ahead_mask = torch.cat((past_mask, look_ahead_mask), dim=1)

    look_ahead_mask *= -1e9

    return look_ahead_mask  # (size, size)

class TimeSteppingModel(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        num_heads: int,
        embed_hidden_dim: int,
        num_layers: int,
        input_seq_len: int,
        output_seq_len: int,
        pars_encoder: ParameterEncoder,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embed_hidden_dim = embed_hidden_dim
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.pars_encoder = pars_encoder

        self.initial_conv_layer = nn.Linear(
            in_features=latent_dim,
            out_features=embed_dim,
            bias=True
        )

        self.positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=input_seq_len + output_seq_len
        )
        
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    embed_hidden_dim=embed_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_conv_layer = nn.Linear(
            in_features=embed_dim,
            out_features=latent_dim,
            bias=True,
        )
    
    def encode_pars(self, pars: torch.Tensor) -> torch.Tensor:
        return self.pars_encoder(pars)

    def decode_next_step(
        self,
        x: torch.Tensor,
        encoded_pars: torch.Tensor,
        ) -> torch.Tensor:

        x = self.initial_conv_layer(x)
        x = self.positional_embedding(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoded_pars)

        x = self.final_conv_layer(x)
        x = x[:, -1:, :]

        return x

    def forward(
        self, 
        x: torch.Tensor,
        pars: torch.Tensor,
        ) -> torch.Tensor:
        pars = self.encode_pars(pars)

        x = self.decode_next_step(x, pars)
        
        return x

    def multistep_prediction(
        self,
        x: torch.Tensor,
        pars: torch.Tensor,
        output_seq_len: int,
        ) -> torch.Tensor:

        input_seq_len = x.shape[1]

        pars = self.encode_pars(pars)

        out = x
        for _ in range(output_seq_len):
            x = self.decode_next_step(out[:, -input_seq_len:], pars)
            out = torch.cat([out, x], dim=1)
        
        return out[:, -output_seq_len:]

    def multistep_prediction_with_teacher_forcing(
        self,
        x: torch.Tensor,
        pars: torch.Tensor,
        output_states: torch.Tensor,
        ) -> torch.Tensor:


        input_seq_len = x.shape[1]
        output_seq_len = output_states.shape[1]

        mask = create_look_ahead_mask(
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            device=self.device
        )

        pars = self.encode_pars(pars)

        out = torch.zeros(
            (x.shape[0], output_seq_len, x.shape[2]),
            device=self.device
        )
        in_out_state = torch.cat([x, output_states], dim=1)
        for i in range(output_seq_len):
            x = self.decode_next_step(in_out_state[:, i:(i+input_seq_len)], pars)
            out[:, i] = x[:, 0]

        return out

    
    @property
    def device(self):
        return next(self.parameters()).device




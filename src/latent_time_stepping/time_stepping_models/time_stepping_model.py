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
    output_seq_len,
    input_seq_len,  
    device='cpu'
    ):
    total_seq_len = output_seq_len + input_seq_len
    mask = torch.ones((total_seq_len, total_seq_len), device=device)
    future_mask = torch.triu(mask, diagonal=1)
    past_mask = torch.tril(mask, diagonal=-input_seq_len)
    mask = future_mask + past_mask
    mask[0:input_seq_len, 0:input_seq_len] = 0

    mask *= -1e9

    return mask  # (size, size)

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
    
    def decode_sequence(
        self,
        x: torch.Tensor,
        encoded_pars: torch.Tensor,
        mask: torch.Tensor = None,
        ) -> torch.Tensor:

        x = self.initial_conv_layer(x)
        x = self.positional_embedding(x)
        
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoded_pars, mask)
        x = self.final_conv_layer(x)

        return x

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
            output_seq_len=output_seq_len-1,
            device=self.device
        )
        pdb.set_trace()

        pars = self.encode_pars(pars)
        
        x = torch.cat([x, output_states[:, 0:-1]], dim=1)
        out = self.decode_sequence(x, pars, mask)

        out = out[:, -output_seq_len:]

        return out

    
    @property
    def device(self):
        return next(self.parameters()).device




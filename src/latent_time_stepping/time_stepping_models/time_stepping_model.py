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
    device='cpu'
    ):
    total_seq_len = input_seq_len
    mask = torch.ones((total_seq_len, total_seq_len), device=device)
    #future_mask = torch.triu(mask, diagonal=1)
    mask = torch.triu(mask, diagonal=1)
    #mask = future_mask + past_mask
    #mask[0:input_seq_len, 0:input_seq_len] = 0
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
        max_seq_len: int,
        pars_encoder: ParameterEncoder,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embed_hidden_dim = embed_hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pars_encoder = pars_encoder

        self.initial_conv_layer = nn.Linear(
            in_features=latent_dim,
            out_features=embed_dim,
            bias=True
        )

        self.positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=max_seq_len + 1
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
        
    def decode_multiple_masked_steps(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        ) -> torch.Tensor:
        
        out = x
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(out, mask)
        out = self.final_conv_layer(out)

        return out[:, -self.max_seq_len:]
    
    def decode_one_step(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:

        out = x
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(out)
        out = self.final_conv_layer(out)
        
        return out[:, -1:]


    def forward(
        self, 
        x: torch.Tensor,
        pars: torch.Tensor,
        ) -> torch.Tensor:
        pars = self.encode_pars(pars)

        x = self.decode(x, pars)
        
        return x

    def multistep_prediction(
        self,
        x: torch.Tensor,
        pars: torch.Tensor,
        output_seq_len: int,
        ) -> torch.Tensor:

        pars = self.encode_pars(pars)

        out = x
        for _ in range(output_seq_len):
            inp = self.initial_conv_layer(out[:, -self.max_seq_len:])
            inp = torch.cat([pars, inp], dim=1)
            inp = self.positional_embedding(inp)

            x = self.decode_one_step(inp)

            x = out[:, -1:] + x

            out = torch.cat([out, x], dim=1)

        return out

    def masked_prediction(
        self,
        x: torch.Tensor,
        pars: torch.Tensor,
        ) -> torch.Tensor:


        input_seq_len = x.shape[1]

        mask = create_look_ahead_mask(
            input_seq_len=input_seq_len+1,
            device=self.device
        )

        pars = self.encode_pars(pars)
        inp = self.initial_conv_layer(x)

        inp = torch.cat([pars, inp], dim=1)

        inp = self.positional_embedding(inp)
        
        out = self.decode_multiple_masked_steps(inp, mask)

        out = x[:, -input_seq_len:] + out[:, -input_seq_len:]

        return out

    
    @property
    def device(self):
        return next(self.parameters()).device




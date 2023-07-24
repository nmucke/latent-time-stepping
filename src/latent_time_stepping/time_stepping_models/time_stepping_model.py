import math
import pdb
from typing import Any, Mapping
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
    total_seq_len = input_seq_len + output_seq_len
    mask = torch.zeros((total_seq_len, total_seq_len), device=device)
    future_mask = torch.ones((output_seq_len, output_seq_len), device=device)
    future_mask = torch.triu(future_mask, diagonal=0)
    #mask = future_mask + past_mask
    #mask[0:input_seq_len, 0:input_seq_len] = 0
    #mask *= -1e9

    mask[input_seq_len:, input_seq_len:] = future_mask
    mask[:input_seq_len, -output_seq_len:] = 1
    mask *= -1e9

    return mask  # (size, size)

def normal_init(m, mean=0., std=1.):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, 1. / math.sqrt(m.weight.size(1)))#std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


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
        self.max_seq_len = input_seq_len + output_seq_len
        self.pars_encoder = pars_encoder

        self.initial_conv_layer = nn.Linear(
            in_features=latent_dim,
            out_features=embed_dim,
            bias=True
        )

        self.positional_embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            max_len=self.max_seq_len + 1
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

        self.apply(normal_init)

    def encode_pars(self, pars: torch.Tensor) -> torch.Tensor:
        return self.pars_encoder(pars)
        
    def decode_multiple_masked_steps(
        self,
        input_output_seq: torch.Tensor,
        output_seq_len: int,
        mask: torch.Tensor = None,
        ) -> torch.Tensor:
        
        out = input_output_seq
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(out, mask)
        out = self.final_conv_layer(out)

        return out[:, -output_seq_len:]
    
    def decode_one_step(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:

        mask = create_look_ahead_mask(
            input_seq_len=x.shape[1],
            output_seq_len=1,
            device=self.device
        )

        padding = torch.zeros((x.shape[0], 1, x.shape[-1]), device=self.device)
        out = torch.cat([x, padding], dim=1)
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(out, mask)
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
        input: torch.Tensor,
        pars: torch.Tensor,
        output_seq_len: int,
        ) -> torch.Tensor:

        input = input.permute(0, 2, 1)
        
        pars = self.encode_pars(pars)

        out = input
        for _ in range(output_seq_len):
            inp = self.initial_conv_layer(out[:, -self.max_seq_len:])
            inp = torch.cat([pars, inp], dim=1)
            inp = self.positional_embedding(inp)

            x = self.decode_one_step(inp)

            #x = out[:, -1:] + x

            out = torch.cat([out, x], dim=1)
        
        out = out[:, -output_seq_len:]
        
        out = out.permute(0, 2, 1)

        return out
    
    def masked_prediction(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        pars: torch.Tensor,
        ) -> torch.Tensor:

        input = input.permute(0, 2, 1)
        output = output.permute(0, 2, 1)

        input_seq_len = input.shape[1]
        output_seq_len = output.shape[1]
        
        mask = create_look_ahead_mask(
            input_seq_len=input_seq_len+1,
            output_seq_len=output_seq_len,
            device=self.device
        )

        pars = self.encode_pars(pars)

        input_output_seq = torch.cat([input, output], dim=1)
        input_output_seq = self.initial_conv_layer(input_output_seq)

        input_output_seq = torch.cat([pars, input_output_seq], dim=1)

        input_output_seq = self.positional_embedding(input_output_seq)
        
        out = self.decode_multiple_masked_steps(
            input_output_seq=input_output_seq, 
            output_seq_len=output_seq_len,
            mask=mask
            )
        out = out[:, -output_seq_len:]

        #out = x[:, -input_seq_len:] + out[:, -input_seq_len:]
        out = out.permute(0, 2, 1)

        return out

    
    @property
    def device(self):
        return next(self.parameters()).device




import math
import numpy as np
import torch
from torch import nn
import pdb
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision

from latent_time_stepping.AE_models.ViT import ViTEncoderLayer

def get_activation_function(activation_function: str) -> nn.Module:

    if activation_function == "relu":
        return nn.ReLU()
    elif activation_function == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_function == "tanh":
        return nn.Tanh()
    elif activation_function == "sigmoid":
        return nn.Sigmoid()
    elif activation_function == "gelu":
        return nn.GELU()
    elif activation_function == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Activation function {activation_function} not supported.")

class LinearPadding(nn.Module):
    def __init__(self, padding: int) -> None:
        super().__init__()

        self.padding = padding
    
    def _right_linear_extrapolation(self, x: torch.Tensor) -> torch.Tensor:

        dx = 1
        
        slope = (x[:, :, -1] - x[:, :, -2]) / dx 
        bias = x[:, :, -2]

        pad = torch.arange(2, self.padding+2, device=x.device)*dx
        pad = pad.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1)

        #pad = torch.zeros(x.shape[0], x.shape[1], self.padding, device=x.device)
        #pad[:, :, 0] = 2*dx

        padding_values = slope[:, :, None] * pad + bias[:, :, None]

        return padding_values
    
    def _left_linear_extrapolation(self, x: torch.Tensor) -> torch.Tensor:

        dx = 1
        
        slope = (x[:, :, 0] - x[:, :, 1]) / dx 
        bias = x[:, :, 1]

        pad = torch.arange(self.padding+2, 2, -1, device=x.device)*dx
        pad = pad.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1)

        padding_values = slope[:, :, None] * pad + bias[:, :, None]

        return padding_values


    def forward(self, x):

        padding_left = self._left_linear_extrapolation(x)
        padding_right = self._right_linear_extrapolation(x)

        x = torch.cat([padding_left, x, padding_right], dim=-1)
        
        return x

class ConvResNetBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        ) -> None:
        super().__init__()

        self.skip_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        x = self.skip_conv(x)

        return out + x

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, seq_len=1000):
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(1,2) + self.pe[:, :x.size(2)]
        return x.transpose(1,2)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input

        self.softmax = nn.Softmax(dim=-1)

        # Embedding dimension of model is a multiple of number of heads

        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads

        # These are still of dimension d_model. To split into number of heads
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesn't saturate
        Q = Q / np.sqrt(self.d_k)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))  # (bs, n_heads, q_length, k_length)

        A = self.softmax(scores)  # (bs, n_heads, q_length, k_length)

        # Get the weighted average of the values
        H = torch.matmul(A, V)  # (bs, n_heads, q_length, dim_per_head)

        return H, A

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()

        # After transforming, split into num_heads
        Q = self.split_heads(self.W_q(X_q), batch_size)
        K = self.split_heads(self.W_k(X_k), batch_size)
        V = self.split_heads(self.W_v(X_v), batch_size)

        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)

        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)  # (bs, q_length, dim)

        # Final linear layer
        H = self.W_h(H_cat)  # (bs, q_length, dim)
        return H, A

class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.k1convL1 = nn.Linear(input_dim, hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            embed_hidden_dim,
            p=0.1
    ):
        super().__init__()

        self.activation = nn.GELU()

        self.mha = MultiHeadAttention(embed_dim, num_heads, p)

        self.feed_forward = FeedForward(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=embed_hidden_dim
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, x):

        # Multi-head attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x)  # (batch_size, seq_len, embed_dim)

        x = self.layernorm1(x + attn_output) # (batch_size, seq_len, embed_dim)

        # Compute accross embedding dimension
        x_ff = self.feed_forward(x)  # (batch_size, seq_len, embed_dim)

        x_ff = self.activation(x_ff)

        x = self.layernorm2(x + x_ff)

        return x

class DecoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            embed_hidden_dim,
            p=0.1
    ):
        super().__init__()

        self.activation = nn.GELU()

        self.mha = MultiHeadAttention(embed_dim, num_heads, p)

        self.feed_forward = FeedForward(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=embed_hidden_dim
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, x, encoder_output):

        # Multi-head self attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x)  # (batch_size, seq_len, embed_dim)

        # Layer norm after adding the residual connection
        x = self.layernorm1(x + attn_output)  # (batch_size, seq_len, embed_dim)

        # Multi-head cross attention
        attn_output, _ = self.mha(X_q=x, X_k=encoder_output, X_v=encoder_output)  # (batch_size, seq_len, embed_dim)

        # Layer norm after adding the residual connection
        x = self.layernorm2(x + attn_output)  # (batch_size, eq_len, embed_dim)

        # Compute accross embedding dimension
        x_ff = self.feed_forward(x)  # (batch_size, seq_len, embed_dim)

        # Layer norm after adding the residual connection
        x = self.layernorm3(x + x_ff)
        return x

def normalization(channels: int):
    return nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6)

class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalization(channels)

        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)

        self.proj_out = nn.Conv1d(channels, channels, 1)

        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):

        x_norm = self.norm(x)

        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)

        out = self.proj_out(out)

        return x + out
  
class UpSample(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        transposed: bool = True,
        ) -> None:
        
        super(UpSample, self).__init__()

        self.padding = kernel_size // 2
        self.transposed = transposed
        
        if self.transposed:

            self.conv = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=self.padding,
                output_padding=1,
            )

        else:

            self.conv = nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=1,
                #padding=self.padding,
                )


        self.linear_padding = LinearPadding(padding=self.padding)

        self.edge_padding = torchvision.transforms.Pad(padding=[self.padding, 0], padding_mode="edge")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if not self.transposed:
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            #x = self.linear_padding(x)
            #x = nn.functional.pad(x, (self.padding, self.padding), mode="constant", value=0)
            x = self.edge_padding(x)

        return self.conv(x)

class DownSample(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding_type: str = 'linear',
        ) -> None:
        
        super(DownSample, self).__init__()

        self.padding_type = padding_type
        
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=0)

        self.linear_padding = LinearPadding(padding=self.padding)
        self.edge_padding = torchvision.transforms.Pad(padding=[self.padding, 0], padding_mode="edge")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        #x = self.linear_padding(x)
        #x = nn.functional.pad(x, (self.padding, self.padding), mode="constant", value=0)
        x = self.edge_padding(x)

        return self.conv(x)

class Encoder(nn.Module):
    def __init__(
        self,
        num_channels: list = None,
        kernel_size: list = None,
        latent_dim: int = 16,
        space_dim: int = 256,
        resnet: bool = True,
        vit: bool = False,
        activation: str = 'leaky_relu',
        embedding_dim: int = None,
        num_transformer_layers: int = 2,
        ):
        super().__init__()

        self.resnet = resnet
        self.vit = vit

        self.embedding_dim = embedding_dim
        
        if num_channels is None:
            num_channels = [2, 32, 64, 128]
        self.num_channels = num_channels

        self.num_layers = len(self.num_channels)

        if kernel_size is None:
            kernel_size = 3
        self.kernel_size = kernel_size

        self.padding = self.kernel_size // 2

        final_dim = space_dim//2**(len(self.num_channels)-1)
        space_dims = [final_dim*2**i for i in range(len(self.num_channels))]
        space_dims.reverse()

        self.latent_dim = latent_dim

        self.activation = get_activation_function(activation)

        self.conv_blocks = nn.ModuleList()
        if self.resnet:
            for i in range(len(self.num_channels)-1):
                self.conv_blocks.append(
                    nn.Sequential(
                        ConvResNetBlock(
                            in_channels=self.num_channels[i],
                            out_channels=self.num_channels[i+1],
                            kernel_size=self.kernel_size,
                        ),
                        self.activation,
                        nn.BatchNorm1d(
                            num_features=self.num_channels[i+1],
                        ),
                        DownSample(
                            in_channels=self.num_channels[i+1],
                            out_channels=self.num_channels[i+1],
                            kernel_size=self.kernel_size,
                        ),
                        self.activation,
                    )
                )
        elif self.vit:

            in_patch_sizes = [2**(self.num_layers - i - 1) for i in range(len(self.num_channels)-1)]
            out_patch_sizes = [2**(self.num_layers - i - 2) for i in range(len(self.num_channels)-1)]

            in_patch_sizes = [1 if patch_size < 1 else patch_size for patch_size in in_patch_sizes]
            out_patch_sizes = [1 if patch_size < 1 else patch_size for patch_size in out_patch_sizes]

            for i in range(len(self.num_channels)-1):
                self.conv_blocks.append(
                    nn.Sequential(
                        ViTEncoderLayer(
                            space_dim=space_dims[i], 
                            in_patch_size=in_patch_sizes[i],
                            out_patch_size=out_patch_sizes[i], 
                            embedding_dim=self.embedding_dim[i], 
                            num_transformer_layers=num_transformer_layers,
                            num_heads=8, 
                            mlp_dim=2*self.embedding_dim[i], 
                            in_channels=self.num_channels[i],
                            out_channels=self.num_channels[i+1],
                        ),
                        self.activation,
                    )
                )

        else:
            for i in range(len(self.num_channels)-1):
                self.conv_blocks.append(
                    nn.Sequential(
                        DownSample(
                            in_channels=self.num_channels[i],
                            out_channels=self.num_channels[i+1],
                            kernel_size=self.kernel_size,
                        ),
                        self.activation,
                        nn.BatchNorm1d(
                            num_features=self.num_channels[i+1],
                        ),
                    )
                )
        
        self.flatten = nn.Flatten()

        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(
            nn.Linear(
                in_features=self.num_channels[-1]*final_dim,
                out_features=self.num_channels[-1],
                bias=True,
            )
        )
        self.dense_layers.append(
            nn.Linear(
                in_features=self.num_channels[-1],
                out_features=self.num_channels[-2],
                bias=True,
            )
        )
        self.final_dense_layers = nn.Linear(
                in_features=self.num_channels[-2],
                out_features=self.latent_dim,
                bias=False,
            )


    def _reshape_input(
        self, 
        x: torch.Tensor,
        batch_size: int,
        num_states: int,
        num_x: int,
        num_time_steps: int,        
        ) -> torch.Tensor:


        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size*num_time_steps, num_states, num_x)

        return x


    def forward(self, x):
        
        batch_size = x.shape[0]
        num_states = x.shape[1]
        num_x = x.shape[2]
        num_time_steps = x.shape[3]


        x = self._reshape_input(x, batch_size, num_states, num_x, num_time_steps)

        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.flatten(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.activation(x)
        
        x = self.final_dense_layers(x)
        
        x = x.reshape(batch_size, num_time_steps, self.latent_dim)
        x = x.permute(0, 2, 1)

        return x

class Decoder(nn.Module):

    def __init__(
        self,
        num_channels: list = None,
        kernel_size: list = None,
        num_states: int = 2,
        latent_dim: int = 16,
        pars_dim: int = 2,    
        space_dim: int = 256,
        resnet: bool = True,
        transposed: bool = False,
        activation: str = 'leaky_relu',
        vit: bool = False,
        embedding_dim: int = None,
        num_transformer_layers: int = 2,
        padding_mode: str = 'edge',
        ):
        super().__init__()

        pars_embed_channels = 8
        init_dim = 4
        self.resnet = resnet
        self.transposed = transposed
        self.vit = vit
        self.pars_dim = pars_dim
        self.padding_mode = padding_mode

        self.embedding_dim = embedding_dim
        
        if num_channels is None:
            num_channels = [128, 64, 32, num_states]
        self.num_channels = num_channels

        self.num_layers = len(self.num_channels)

        if kernel_size is None:
            kernel_size = 3
        self.kernel_size = kernel_size

        self.padding = self.kernel_size//2

        self.latent_dim = latent_dim
        self.activation = get_activation_function(activation)

        init_dim = space_dim//2**(len(self.num_channels)-1)
        space_dims = [init_dim*2**i for i in range(len(self.num_channels))]

        self.latent_dim = latent_dim

        self.init_dense_layer = nn.Linear(
                in_features=self.latent_dim,
                out_features=self.num_channels[1],
                bias=True,
            )

        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(
            nn.Linear(
                in_features=self.num_channels[1],
                out_features=self.num_channels[0],
                bias=True,
            )
        )
        self.dense_layers.append(
            nn.Linear(
                in_features=self.num_channels[0],
                out_features=self.num_channels[0]*init_dim,
                bias=True,
            )
        )

        if self.pars_dim is not None:
            self.pars_embed_1 = nn.Linear(
                in_features=pars_dim,
                out_features=pars_embed_channels,
                bias=True,
            )
            self.pars_embed_2 = nn.Linear(
                in_features=pars_embed_channels,
                out_features=pars_embed_channels * init_dim,# if not self.vit else self.num_channels[0] * init_dim,
                bias=True,
            )
            
            self.pars_unflatten = nn.Unflatten(1, (pars_embed_channels, init_dim))

        self.state_unflatten = nn.Unflatten(1, (self.num_channels[0], init_dim))

        if self.pars_dim is not None:
            self.num_channels[0] += pars_embed_channels
        

        self.conv_blocks = nn.ModuleList()
        if self.resnet:
            for i in range(len(self.num_channels)-1):
                self.conv_blocks.append(
                    nn.Sequential(
                        ConvResNetBlock(
                            in_channels=self.num_channels[i],
                            out_channels=self.num_channels[i+1],
                            kernel_size=self.kernel_size,
                        ),
                        self.activation,
                        nn.BatchNorm1d(
                            num_features=self.num_channels[i+1],
                        ),
                        UpSample(
                            in_channels=self.num_channels[i+1],
                            out_channels=self.num_channels[i+1],
                            kernel_size=self.kernel_size,
                            transposed=self.transposed,
                        ),
                        self.activation,
                    )
                )
        elif self.vit:

            in_patch_sizes = [2**(i) for i in range(len(self.num_channels)-1)]
            out_patch_sizes = [2**(i+1) for i in range(len(self.num_channels)-1)]

            in_patch_sizes = [1 if patch_size < 1 else patch_size for patch_size in in_patch_sizes]
            out_patch_sizes = [1 if patch_size < 1 else patch_size for patch_size in out_patch_sizes]

            for i in range(len(self.num_channels)-1):
                self.conv_blocks.append(
                    nn.Sequential(
                        ViTEncoderLayer(
                            space_dim=space_dims[i], 
                            in_patch_size=in_patch_sizes[i],
                            out_patch_size=out_patch_sizes[i], 
                            embedding_dim=self.embedding_dim[i], 
                            num_transformer_layers=num_transformer_layers, 
                            num_heads=8, 
                            mlp_dim=2*self.embedding_dim[i], 
                            in_channels=self.num_channels[i],
                            out_channels=self.num_channels[i+1],
                        ),
                        self.activation,
                    )
                )
        else:
            for i in range(len(self.num_channels)-1):
                self.conv_blocks.append(
                    nn.Sequential(
                        UpSample(
                            in_channels=self.num_channels[i],
                            out_channels=self.num_channels[i+1],
                            kernel_size=self.kernel_size,
                            transposed=self.transposed,
                        ),
                        self.activation,
                        nn.BatchNorm1d(
                            num_features=self.num_channels[i+1],
                        ),
                    )
                )
        

        self.edge_padding = torchvision.transforms.Pad(padding=[self.padding, 0], padding_mode="edge")
        #if self.padding_mode == "edge":
        #    self.edge_padding = torchvision.transforms.Pad(padding=[self.padding, 0], padding_mode="edge")
        #elif self.padding_mode == "periodic":
        #    self.edge_padding = torchvision.transforms.Pad(padding=[self.padding, 0], padding_mode="circular")

        self.final_conv = nn.Conv1d(
            in_channels=self.num_channels[-1],
            out_channels=self.num_channels[-1],
            kernel_size=self.kernel_size,
            bias=False,
        )


    def _reshape_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        latent_dim: int,
        num_time_steps: int,
    ) -> torch.Tensor:

        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size*num_time_steps, latent_dim)

        return x

    def forward(self, x, pars=None):

        batch_size = x.shape[0]
        latent_dim = x.shape[1]
        num_time_steps = x.shape[2]

        x = self._reshape_input(x, batch_size, latent_dim, num_time_steps)
        
        if self.pars_dim is not None:
            pars = self.pars_embed_1(pars)
            pars = self.activation(pars)
            pars = self.pars_embed_2(pars)
            pars = self.activation(pars)

            pars = self.pars_unflatten(pars)
            
        x = self.init_dense_layer(x)
        x = self.activation(x)
        for dense in self.dense_layers:
            x = dense(x)
            x = self.activation(x)

        x = self.state_unflatten(x)

        if self.pars_dim is not None:
            pars = torch.repeat_interleave(pars, repeats=num_time_steps, dim=0)
            x = torch.cat([x, pars], dim=1)

        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        #if not self.vit:
        x = self.edge_padding(x)
        x = self.final_conv(x)

        x = x.reshape(batch_size, num_time_steps, self.num_channels[-1], x.shape[-1])
        x = x.permute(0, 2, 3, 1)
        
        return x
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm
import time
import math
import torch.nn.functional as F



class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        k_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesn't saturate
        Q = Q / np.sqrt(self.d_k)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))  # (bs, n_heads, q_length, k_length)

        # Masking
        if mask is not None:
            scores += mask 

        A = self.softmax(scores)  # (bs, n_heads, q_length, k_length)

        # Get the weighted average of the values
        H = torch.matmul(A, V)  # (bs, n_heads, q_length, dim_per_head)

        return H, A

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    def forward(self, X_q, X_k, X_v, mask=None):
        batch_size, seq_length, dim = X_q.size()

        # After transforming, split into num_heads
        Q = self.split_heads(self.W_q(X_q), batch_size)
        K = self.split_heads(self.W_k(X_k), batch_size)
        V = self.split_heads(self.W_v(X_v), batch_size)

        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)  # (bs, q_length, dim)

        # Final linear layer
        H = self.W_h(H_cat)  # (bs, q_length, dim)
        return H, A


class Feedforward(nn.Module):
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
        self.feedforward = Feedforward(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=embed_hidden_dim
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x)  # (batch_size, input_seq_len, input_embed_dim)

        # Layer norm after adding the residual connection
        x = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, input_embed_dim)

        # Compute accross embedding dimension
        x_ff = self.feedforward(x)  # (batch_size, input_seq_len, output_embed_dim)
        x = self.layernorm2(x_ff + x)

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
        self.feedforward = Feedforward(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=embed_hidden_dim
        )


        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        #self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

    def forward(self, x, mask=None):

        x = self.layernorm1(x)  # (batch_size, input_seq_len, input_embed_dim)

        # Multi-head self attention
        attn_output, _ = self.mha(X_q=x, X_k=x, X_v=x, mask=mask)  # (batch_size, input_seq_len, input_embed_dim)

        x = x + attn_output
        
        x = self.layernorm3(x)

        # Compute accross embedding dimension
        x_ff = self.feedforward(x)  # (batch_size, input_seq_len, output_embed_dim)

        x = x_ff + x

        return x

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()



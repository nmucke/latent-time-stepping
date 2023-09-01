
'''
Code coming from https://github.com/lucidrains/vit-pytorch
'''



import pdb
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()

        assert seq_len % patch_size == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, series):
        *_, n, dtype = *series.shape, series.dtype

        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)


        # Rearrange back to (batch, seq_len, channels)
        #x = rearrange(x, 'b n d -> b n () d')



        x = x.mean(dim = 1)


        x = self.to_latent(x)
        return self.linear_head(x)
    
class ViTEncoderLayer(nn.Module):
    def __init__(
        self, 
        space_dim, 
        in_patch_size, 
        out_patch_size,
        embedding_dim, 
        num_transformer_layers, 
        num_heads, 
        mlp_dim, 
        in_channels = 2, 
        out_channels = 4
        ) -> None:
        super().__init__()

        assert space_dim % in_patch_size == 0

        num_patches = space_dim // in_patch_size
        in_patch_dim = in_channels * in_patch_size

        out_patch_dim = out_channels * out_patch_size

        print(f'num_patches: {num_patches}')
        print(f'in_patch_dim: {in_patch_dim}, in_patch_size: {in_patch_size}')
        print(f'out_patch_dim: {out_patch_dim}, out_patch_size: {out_patch_size}')
        print(f'embedding_dim: {embedding_dim}')
        print('------------------------------------------')


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = in_patch_size),
            nn.LayerNorm(in_patch_dim),
            nn.Linear(in_patch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        #self.transformer = Transformer(embedding_dim, num_transformer_layers, num_heads, embedding_dim, mlp_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                batch_first=True,
                ), 
            num_layers=num_transformer_layers,
        )

        self.project_from_patch = nn.Sequential(
            nn.Linear(embedding_dim, out_patch_dim),
            nn.LayerNorm(out_patch_dim),
            Rearrange('b n (p c) -> b c (n p)', p = out_patch_size)
        )

        #self.activation = nn.LeakyReLU()

        
        #self.conv = nn.Conv1d(out_channels, out_channels, kernel_size = 5, padding = 2)

    def forward(self, series):
        #*_, n, dtype = *series.shape, series.dtype

        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe


        x = self.transformer(x)
        x = self.project_from_patch(x)
        #x = self.activation(x)
        #x = self.conv(x)
        return x
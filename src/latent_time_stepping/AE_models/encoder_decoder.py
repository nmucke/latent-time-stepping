import numpy as np
import torch
from torch import nn
import pdb


def normalization(channels: int):
    return nn.GroupNorm(num_groups=4, num_channels=channels, eps=1e-6)


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
        attn = nn.functional.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)

        out = self.proj_out(out)

        return x + out

            
        
class UpSample(nn.Module):

    def __init__(
        self,
        channels: int,
        ) -> None:
        
        super(UpSample, self).__init__()
        
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        
        return self.conv(x)

class DownSample(nn.Module):

    def __init__(
        self,
        channels: int,
        ) -> None:
        
        super(DownSample, self).__init__()
        
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = nn.functional.pad(x, (0, 1), mode="replicate")

        return self.conv(x)

class Encoder(nn.Module):
    def __init__(
        self,
        num_channels: list = None,
        kernel_size: list = None,
        latent_dim: int = 16,
        stride: list = None,
        padding: list = None,
        ):
        super().__init__()
        
        if num_channels is None:
            num_channels = [2, 32, 64, 128]
        self.num_channels = num_channels

        if kernel_size is None:
            kernel_size = [3, 3, 3, 3]
        self.kernel_size = kernel_size

        self.latent_dim = latent_dim

        self.activation = nn.LeakyReLU()


        self.conv_list = nn.ModuleList()
        for i in range(len(self.num_channels)-1):
            self.conv_list.append(
                nn.Conv1d(
                    in_channels=self.num_channels[i],
                    out_channels=self.num_channels[i+1],
                    kernel_size=self.kernel_size,
                    bias=True,
                    stride=1,
                    padding=1,
                )
            )
        
        self.down = nn.ModuleList()
        for i in range(len(self.num_channels)-1):
            self.down.append(
                DownSample(
                    channels=self.num_channels[i+1],
                )
            )
            
        out_size = 256//2**(len(self.num_channels)-1)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=self.num_channels[-1]*out_size,
            out_features=self.latent_dim,
            bias=False
        )

        self.attention = AttnBlock(self.num_channels[-1])

    def forward(self, x):
        for (conv, down) in zip(self.conv_list, self.down):
            x = conv(x)
            x = self.activation(x)
            x = down(x)
        x = self.activation(x)
        x = self.attention(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.activation(x)
        x = self.fc1(x)
        return x

class Decoder(nn.Module):

    def __init__(
        self,
        num_channels: list = None,
        kernel_size: list = None,
        latent_dim: int = 16,
        stride: list = None,
        padding: list = None,
        output_padding: list = None,        
        ):
        super().__init__()

        init_dim = 8
        
        if num_channels is None:
            num_channels = [128, 64, 32, 2]
        self.num_channels = num_channels

        if kernel_size is None:
            kernel_size = [3, 3, 3, 3]
        self.kernel_size = kernel_size

        self.latent_dim = latent_dim
        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.num_channels[0]*init_dim,
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(self.num_channels[0], init_dim),
        )

        
        self.conv_list = nn.ModuleList()
        for i in range(len(self.num_channels) - 1):
            self.conv_list.append(
                nn.Conv1d(
                    in_channels=self.num_channels[i],
                    out_channels=self.num_channels[i+1],
                    kernel_size=self.kernel_size,
                    bias=True,
                    stride=stride,
                    padding=padding,
                    #output_padding=output_padding[i],
                )
            )
        
        self.up_layers = nn.ModuleList()
        for i in range(len(self.num_channels) - 1):
            self.up_layers.append(
                UpSample(
                    channels=self.num_channels[i+1],
                )
            )
        
        self.output_conv = nn.Conv1d(
            in_channels=self.num_channels[-1],
            out_channels=self.num_channels[-1],
            kernel_size=self.kernel_size,
            bias=False,
            stride=stride,
            padding=padding,
        )

        self.attention = AttnBlock(self.num_channels[0])
                
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.unflatten(x)
        x = self.attention(x)
        x = self.activation(x)
        for (conv, up) in zip(self.conv_list, self.up_layers):
            x = conv(x)
            x = self.activation(x)
            x = up(x)
        x = self.activation(x)
        x = self.output_conv(x)
        return x
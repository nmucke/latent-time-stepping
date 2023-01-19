import torch
from torch import nn
import pdb

def conv1d_output_shape(
    input_shape: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ) -> int:
    """Calculate output shape of 1D convolutional layer"""
    
    output_shape = \
        (input_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    
    return int(output_shape)

def get_final_conv1d_output_shape(
    input_shape: int,
    kernel_size: list,
    stride: list,
    padding: list,
    dilation: list,
    ) -> int:
    """Calculate output shape of 1D convolutional layer"""
    output_shape = input_shape
    for i in range(len(kernel_size)):
        output_shape = \
            conv1d_output_shape(
                input_shape=output_shape,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=dilation[i],
                )
            
    return output_shape

def conv1d_transpose_output_shape(
    input_shape: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    output_padding: int,
    ) -> int:
    """Calculate output shape of 1D convolutional transpose layer"""
    
    output_shape = \
        (input_shape - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + \
        output_padding + 1
    
    return int(output_shape)

def get_final_conv1d_transpose_output_shape(
    input_shape: int,
    kernel_size: list,
    stride: list,
    padding: list,
    dilation: list,
    output_padding: list,
    ) -> int:
    """Calculate output shape of 1D convolutional transpose layer"""

    output_shape = input_shape
    for i in range(len(kernel_size)):
        output_shape = \
            conv1d_transpose_output_shape(
                input_shape=output_shape,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=dilation[i],
                output_padding=output_padding[i],
                )

    return output_shape

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
                    kernel_size=self.kernel_size[i],
                    bias=True,
                    stride=stride[i],
                    padding=padding[i],
                )
            )
        
        self.flatten = nn.Flatten()

        self.output_shape = get_final_conv1d_output_shape(
            input_shape=256,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=[1]*len(self.kernel_size),
        )

        self.fc1 = nn.Linear(
            in_features=self.num_channels[-1]*self.output_shape,
            out_features=self.num_channels[-1],
            bias=True
        )
        self.fc2 = nn.Linear(
            in_features=self.num_channels[-1],
            out_features=self.latent_dim,
            bias=False
        )

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
            x = self.activation(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
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
            out_features=self.num_channels[0],
        )

        self.fc2 = nn.Linear(
            in_features=self.num_channels[0],
            out_features=self.num_channels[0]*5,
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(self.num_channels[0], 5),
        )
        
        self.conv_list = nn.ModuleList()
        for i in range(len(self.num_channels) - 1):
            self.conv_list.append(
                nn.ConvTranspose1d(
                    in_channels=self.num_channels[i],
                    out_channels=self.num_channels[i+1],
                    kernel_size=self.kernel_size[i],
                    bias=True,
                    stride=stride[i],
                    padding=padding[i],
                    output_padding=output_padding[i],
                )
            )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.unflatten(x)
        for conv in self.conv_list:
            x = conv(x)
            x = self.activation(x)
        
        return x
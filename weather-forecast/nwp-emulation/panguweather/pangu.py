# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
from typing import Any

import numpy as np
import torch
from timm.layers import DropPath
from torch import Tensor, nn
from torch.nn import (
    GELU,
    ConstantPad2d,
    ConstantPad3d,
    Conv2d,
    Conv3d,
    ConvTranspose2d,
    ConvTranspose3d,
    Dropout,
    LayerNorm,
    Linear,
    Softmax,
)
from torch.utils.checkpoint import checkpoint

from utils import (
    cat_constant_masks,
    define_3d_earth_position_index,
    generate_3d_attention_mask,
)


class PanguModel(nn.Module):
    """PanguWeather network as described in http://arxiv.org/abs/2211.02556
    and https://www.nature.com/articles/s41586-023-06185-3.
    This implementation follows the official pseudo code here:
    https://github.com/198808xc/Pangu-Weather

        Args:
            plevel_patch_size (tuple[int]): Patch size for the pressure level
            data. Default is (2, 4, 4). Setting (2, 8, 8) leads to Pangu Lite.
            constant_mask_list (list[str]): list of the constant masks to add
            to the surface data as additional channels (e. g., soil type).
            token_size (int): Size of the tokens (equivalent to channel size)
            of the first layer. Default is 192.
            layer_depth (tuple[int, int]): Number of blocks in layers. Default
            is (2, 6), meaning that the first and fourth layers contain 2
            blocks, and the second and third contain 6.
            num_heads (tuple[int, int]): Number of heads in attention layers.
            Default is (6, 12), corresponding to respectively first and fourth
            layers, and second and third.
    """

    def __init__(
        self,
        latitude: int,
        longitude: int,
        surface_variables: int,
        plevel_variables: int,
        plevels: int,
        plevel_patch_size: tuple[int, int, int] = (2, 4, 4),
        constant_masks: dict[str, dict] | None = None,
        token_size: int = 192,
        layer_depth: tuple[int, int] = (2, 6),
        num_heads: tuple[int, int] = (6, 12),
        add_const2: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__()

        self.plevel_patch_size = plevel_patch_size

        # load constant mask, e. g., soil type
        if constant_masks is not None:
            self.register_buffer(
                "constant_masks",
                torch.stack(
                    [torch.from_numpy(m["mask"]) for m in constant_masks.values()]
                ),
            )
            surface_channels = surface_variables + len(constant_masks)
        else:
            surface_channels = surface_variables
            self.constant_masks = None

        # Special plevel const from onnx version of pangu, level 0 == close to the ground,
        # shape = 1, 1, 1, 13, 721, 1440
        if add_const2:
            self.register_buffer(
                "const2",
                torch.flip(
                    torch.from_numpy(
                        np.load(
                            os.path.join(
                                pathlib.Path(__file__).parent.resolve(), "const2.npy"
                            )
                        ).squeeze(0)
                    ),
                    dims=(2,),
                ),
            )
            plevel_size = torch.Size(
                (plevel_variables + 1, plevels, latitude, longitude)
            )
        else:
            plevel_size = torch.Size((plevel_variables, plevels, latitude, longitude))

        # Set data size. Needed to compute the size of earth specific bias
        surface_size = torch.Size((surface_channels, latitude, longitude))

        # Drop path rate is linearly increased as the depth increases
        drop_path_list = torch.linspace(0, 0.2, 8)

        # Patch embedding
        self.embedding_layer = PatchEmbedding(
            token_size,
            self.plevel_patch_size,
            plevel_size=plevel_size,
            surface_size=surface_size,
        )
        embedding_size = self.embedding_layer.embedding_size

        # Upsample and downsample
        self.downsample = DownSample(embedding_size, token_size)
        downsampled_size = self.downsample.downsampled_size
        self.upsample = UpSample(token_size * 2, token_size)

        # Four Earth specific layers
        self.layer1 = EarthSpecificLayer(
            layer_depth[0],
            embedding_size,
            token_size,
            drop_path_list[:2],
            num_heads[0],
            **kwargs,
        )
        self.layer2 = EarthSpecificLayer(
            layer_depth[1],
            downsampled_size,
            token_size * 2,
            drop_path_list[2:],
            num_heads[1],
            **kwargs,
        )
        self.layer3 = EarthSpecificLayer(
            layer_depth[1],
            downsampled_size,
            token_size * 2,
            drop_path_list[2:],
            num_heads[1],
            **kwargs,
        )
        self.layer4 = EarthSpecificLayer(
            layer_depth[0],
            embedding_size,
            token_size,
            drop_path_list[:2],
            num_heads[0],
            **kwargs,
        )

        # Patch Recovery
        self._output_layer = PatchRecovery(token_size * 2, self.plevel_patch_size)

    def forward(
        self, input_plevel: Tensor, input_surface: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Add constant masks to surface data if any
        if self.constant_masks is not None:
            surface_data = cat_constant_masks(input_surface, self.constant_masks)
        else:
            surface_data = input_surface

        # Concat the special const2
        if hasattr(self, "const2"):
            assert isinstance(self.const2, Tensor), "const2 is not a tensor"
            input_plevel = torch.cat(
                [
                    input_plevel,
                    self.const2.expand(
                        (input_plevel.shape[0],) + tuple(self.const2.shape[1:])
                    ),
                ],
                dim=1,
            )

        # Embed the input fields into patches
        x, embedding_shape = self.embedding_layer(input_plevel, surface_data)

        # Encoder, composed of two layers
        x = self.layer1(x, embedding_shape)

        # Store the tensor for skip-connection
        skip = x

        # Downsample from (8, 181, 360) to (8, 91, 180)
        x, downsampled_shape = self.downsample(x, embedding_shape)

        # Layer 2, shape (8, 91, 180, 2C), C = 192 as in the original paper
        x = self.layer2(x, downsampled_shape)

        # Decoder, composed of two layers
        # Layer 3, shape (8, 91, 180, 2C), C = 192 as in the original paper
        x = self.layer3(x, downsampled_shape)

        # Upsample from (8, 91, 180) to (8, 181, 360)
        x = self.upsample(x, embedding_shape)

        # Layer 4, shape (8, 181, 360, 2C), C = 192 as in the original paper
        x = self.layer4(x, embedding_shape)

        # Skip connect, in last dimension(C from 192 to 384)
        x = torch.cat([x, skip], dim=-1)

        # Recover the output fields from patches
        output_plevel, output_surface = self._output_layer(x, embedding_shape)

        # Crop the output to remove zero-paddings
        padded_z, padded_h, padded_w = output_plevel.shape[2:5]
        (
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            padding_front,
            padding_back,
        ) = self.embedding_layer.pad_plevel_data.padding
        output_plevel = output_plevel[
            :,
            :,
            padding_front : padded_z - padding_back,
            padding_top : padded_h - padding_bottom,
            padding_left : padded_w - padding_right,
        ]
        output_surface = output_surface[
            :,
            :,
            padding_top : padded_h - padding_bottom,
            padding_left : padded_w - padding_right,
        ]

        return output_plevel, output_surface


class CustomPad3d(ConstantPad3d):
    """Custom 3d padding based on token embedding patch size. Padding
    direction is center.

    Args:
        data_size (torch.Size): data size
        patch_size (tuple[int]): patch size for the token embedding operation
        value (float, optional): padding value. Defaults to 0.
    """

    def __init__(
        self,
        data_size: torch.Size,
        patch_size: tuple[int, int, int],
        value: float = 0.0,
    ) -> None:
        # Compute paddings, starts from the last dim and goes backward
        assert (
            len(data_size) == 3
        ), f"This padding class is for 3d data, but data has {len(data_size)} dimension(s)"
        assert (
            len(patch_size) == 3
        ), f"Patch should be 3d, but has {len(patch_size)} dimension(s)"
        padding_lon = (
            patch_size[-1] - (data_size[-1] % patch_size[-1])
            if (data_size[-1] % patch_size[-1]) > 0
            else 0
        )
        padding_left = padding_lon // 2
        padding_right = padding_lon - padding_left
        padding_lat = (
            patch_size[-2] - (data_size[-2] % patch_size[-2])
            if (data_size[-2] % patch_size[-2]) > 0
            else 0
        )
        padding_top = padding_lat // 2
        padding_bottom = padding_lat - padding_top
        padding_level = (
            patch_size[-3] - (data_size[-3] % patch_size[-3])
            if (data_size[-3] % patch_size[-3]) > 0
            else 0
        )
        padding_front = padding_level // 2
        padding_back = padding_level - padding_front
        super().__init__(
            padding=(
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            ),
            value=value,
        )
        self.padded_size: torch.Size = torch.Size(
            [
                data_size[0] + padding_level,
                data_size[1] + padding_lat,
                data_size[2] + padding_lon,
            ]
        )


class CustomPad2d(ConstantPad2d):
    """Custom 2d padding based on token embedding patch size. Padding direction is center.

    Args:
        data_size (torch.Size): data size
        patch_size (tuple[int]): patch size for the token embedding operation
        value (float, optional): padding value. Defaults to 0.
    """

    def __init__(
        self, data_size: torch.Size, patch_size: tuple[int, int], value: float = 0.0
    ) -> None:
        # Compute paddings, starts from the last dim and goes backward
        assert (
            len(data_size) == 2
        ), f"This padding class is for 2d data, but data has {len(data_size)} dimension(s)"
        assert (
            len(patch_size) == 2
        ), f"Patch should be 2d, but has {len(patch_size)} dimension(s)"
        padding_lon = (
            patch_size[-1] - (data_size[-1] % patch_size[-1])
            if (data_size[-1] % patch_size[-1]) > 0
            else 0
        )
        padding_left = padding_lon // 2
        padding_right = padding_lon - padding_left
        padding_lat = (
            patch_size[-2] - (data_size[-2] % patch_size[-2])
            if (data_size[-2] % patch_size[-2]) > 0
            else 0
        )
        padding_top = padding_lat // 2
        padding_bottom = padding_lat - padding_top
        super().__init__(
            padding=(padding_left, padding_right, padding_top, padding_bottom),
            value=value,
        )
        self.padded_size = torch.Size(
            [data_size[0] + padding_lat, data_size[1] + padding_lon]
        )


class PatchEmbedding(nn.Module):
    """Patch embedding operation. Apply a linear projection for
    patch_size[0]*patch_size[1]*patch_size[2] patches,
    patch_size = (2, 4, 4) in the original paper

    Args:
        c_dim (int): embeeding channel size
        patch_size (tuple[int, int, int]): patch size for pressure level data
        plevel_size (torch.Size): pressure level data size
        surface_size (torch.Size): surface data size
    """

    def __init__(
        self,
        c_dim: int,
        patch_size: tuple[int, int, int],
        plevel_size: torch.Size,
        surface_size: torch.Size,
    ) -> None:
        super().__init__()

        # Here we use convolution to partition data into cubes
        self.conv_surface = Conv2d(
            in_channels=surface_size[0],
            out_channels=c_dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )
        self.conv_plevel = Conv3d(
            in_channels=plevel_size[0],
            out_channels=c_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # init padding
        self.pad_plevel_data = CustomPad3d(plevel_size[-3:], patch_size)
        self.pad_surface_data = CustomPad2d(surface_size[-2:], patch_size[1:])

        # Compute output size
        plevel_padded_size = self.pad_plevel_data.padded_size
        embedding_size = [
            plevel_dim // patch_dim
            for plevel_dim, patch_dim in zip(plevel_padded_size, patch_size)
        ]
        embedding_size[0] += 1
        self.embedding_size = torch.Size(embedding_size)

    def forward(
        self, input_plevel: Tensor, input_surface: Tensor
    ) -> tuple[Tensor, torch.Size]:
        # Zero-pad the input
        plevel_data = self.pad_plevel_data(input_plevel)
        surface_data = self.pad_surface_data(input_surface)

        # Project to embedding space
        plevel_tokens = self.conv_plevel(plevel_data)
        surface_tokens = self.conv_surface(surface_data)

        # Concatenate the input in the pressure level, i.e., in Z dimension
        x = torch.cat([plevel_tokens, surface_tokens.unsqueeze(2)], dim=2)

        # Reshape x for calculation of linear projections
        x = x.permute((0, 2, 3, 4, 1))
        embedding_shape = x.shape
        x = x.reshape(shape=(x.shape[0], -1, x.shape[-1]))

        return x, embedding_shape


class PatchRecovery(nn.Module):
    """Patch recovery operation. The inverse operation of the patch embedding
    operation.

    Args:
        dim (int): number of channels
        patch_size (tuple[int, int, int]): pressure level patch size, e. g., (2, 4, 4)
        as in the original paper
        plevel_channels (int, optional): pressure level data channel size
        surface_channels (int, optional): surface data channel size
    """

    def __init__(
        self,
        dim: int,
        patch_size: tuple[int, int, int],
        plevel_channels: int = 5,
        surface_channels: int = 4,
    ) -> None:
        super().__init__()
        # Here, we use two transposed convolutions to recover data
        self.conv_surface = ConvTranspose2d(
            in_channels=dim,
            out_channels=surface_channels,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )
        self.conv = ConvTranspose3d(
            in_channels=dim,
            out_channels=plevel_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor, embedding_shape: torch.Size) -> tuple[Tensor, Tensor]:
        # Reshape x back to three dimensions
        x = x.reshape(
            x.shape[0], embedding_shape[1], embedding_shape[2], embedding_shape[3], -1
        )
        x = x.permute(0, 4, 1, 2, 3)

        # Call the transposed convolution
        output_plevel: Tensor = self.conv(x[:, :, :-1, :, :].contiguous())
        output_surface: Tensor = self.conv_surface(x[:, :, -1, :, :].contiguous())

        return output_plevel, output_surface


class DownSample(nn.Module):
    """Down-sampling operation. The number of tokens is divided by 4 while
    their size in multiplied by 2.
    E. g., from (8x360x181) tokens of size 192 to (8x180x91) tokens of size
    384.

        Args:
            dim (int): initial size of the tokens
    """

    def __init__(self, data_size: torch.Size, dim: int) -> None:
        super().__init__()
        # A linear function and a layer normalization
        self.linear = Linear(in_features=4 * dim, out_features=2 * dim, bias=False)
        self.norm = LayerNorm(normalized_shape=4 * dim)
        self.pad = CustomPad3d(data_size[-3:], (1, 2, 2))
        padded_size = self.pad.padded_size
        self.downsampled_size = torch.Size(
            [padded_size[0], padded_size[1] // 2, padded_size[2] // 2]
        )

    def forward(
        self, x: Tensor, embedding_shape: torch.Size
    ) -> tuple[Tensor, torch.Size]:
        # Reshape x to three dimensions for downsampling
        x = x.reshape(shape=embedding_shape)

        # Padding the input to facilitate downsampling
        x = x.permute((0, 4, 1, 2, 3))
        x = self.pad(x)
        x = x.permute((0, 2, 3, 4, 1))

        # Reorganize x to reduce the resolution: simply change the order and
        # downsample from (8, 182, 360) to (8, 91, 180)
        z, h, w = x.shape[1:4]
        # Reshape x to facilitate downsampling
        x = x.reshape(shape=(x.shape[0], z, h // 2, 2, w // 2, 2, x.shape[-1]))
        # Change the order of x
        x = x.permute((0, 1, 2, 4, 3, 5, 6))
        # Reshape to get a tensor of resolution (8, 91, 180) -> 4 times less
        # tokens of 4 times bigger size
        x = x.reshape(shape=(x.shape[0], z * (h // 2) * (w // 2), -1))

        # Call the layer normalization
        x = self.norm(x)

        # Decrease the size of the tokens to reduce computation cost
        x = self.linear(x)
        return x, torch.Size([x.shape[0], z, (h // 2), (w // 2), x.shape[-1]])


class UpSample(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Up-sampling operation. The number of tokens is mutiplied by 4.

        Args:
            input_dim (int): input token size
            output_dim (int): output token size
        """
        super().__init__()
        # Linear layers without bias to increase channels of the data
        self.linear1 = Linear(input_dim, output_dim * 4, bias=False)

        # Linear layers without bias to mix the data up
        self.linear2 = Linear(output_dim, output_dim, bias=False)

        # Normalization
        self.norm = LayerNorm(output_dim)

    def forward(self, x: Tensor, embedding_shape: torch.Size) -> Tensor:
        assert (
            x.shape[-1] % 4 == 0
        ), f"The token size must be divisible by 4, but is {x.shape[-1]}"
        # Z, H, W represent the desired output shape
        h_d = embedding_shape[2] // 2 + embedding_shape[2] % 2
        w_d = embedding_shape[3] // 2 + embedding_shape[3] % 2

        # Call the linear functions to increase channels of the data
        x = self.linear1(x)

        # Reorganize x to increase the resolution: simply change the order and
        # upsample from (8, 91, 180) to (8, 182, 360)
        # Reshape x to facilitate upsampling.
        x = x.reshape(shape=(x.shape[0], embedding_shape[1], h_d, w_d, 2, 2, -1))
        # Change the order of x
        x = x.permute((0, 1, 2, 4, 3, 5, 6))
        # Reshape to get Tensor with a resolution of (8, 182, 360)
        x = x.reshape(shape=(x.shape[0], embedding_shape[1], h_d * 2, w_d * 2, -1))

        # Crop the output to the input shape of the network
        x = x[:, :, : embedding_shape[2], : embedding_shape[3], :]

        # Reshape x back
        x = x.reshape(shape=(x.shape[0], -1, x.shape[-1]))

        # Call the layer normalization
        x = self.norm(x)

        # Mixup normalized tensors
        x = self.linear2(x)
        return x


class EarthSpecificLayer(nn.Module):
    """Basic layer of our network, contains 2 or 6 blocks

    Args:
        depth (int): number of blocks
        data_size (torch.Size): see EarthSpecificBlock
        dim (int): see EarthSpecificBlock
        drop_path_ratio_list (Tensor): see EarthSpecificBlock
        num_heads (int): see EarthSpecificBlock
    """

    def __init__(
        self,
        depth: int,
        data_size: torch.Size,
        dim: int,
        drop_path_ratio_list: Tensor,
        num_heads: int,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        self.blocks = nn.ModuleList()

        # Construct basic blocks
        for i in range(depth):
            self.blocks.append(
                EarthSpecificBlock(
                    data_size, dim, drop_path_ratio_list[i].item(), num_heads, **kwargs
                )
            )

    def forward(
        self, x: Tensor, embedding_shape: torch.Size, *args: Any, **kwargs: Any
    ) -> Tensor:
        for i, block in enumerate(self.blocks):
            # Roll the input every two blocks
            if i % 2 == 0:
                x = block(x, embedding_shape, roll=False, *args, **kwargs)
            else:
                x = block(x, embedding_shape, roll=True, *args, **kwargs)
        return x


class EarthSpecificBlock(nn.Module):
    """3D transformer block with Earth-Specific bias and window attention,
    see https://github.com/microsoft/Swin-Transformer for the official
    implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the
    relative position bias with Earth-Specific bias.

    Args:
        data_size (torch.Size): data size in terms of plevel, latitude, longitude
        dim (int): token size
        drop_path_ratio (float): ratio to apply to drop path
        num_heads (int): number of attention heads
        window_size (tuple[int], optional): window size for the sliding window
        attention. Defaults to (2, 6, 12).
        dropout_rate (float, optional): dropout rate in the MLP. Defaults to 0..
        checkpoint_activation (bool): whether to use checkpointing for the
        activation function. Defaults to False.
        lam (bool): activates the limited-area model masks. Defaults to False.
    """

    def __init__(
        self,
        data_size: torch.Size,
        dim: int,
        drop_path_ratio: float,
        num_heads: int,
        window_size: tuple[int, int, int] = (2, 6, 12),
        dropout_rate: float = 0.0,
        checkpoint_activation: bool = False,
        lam: bool = False,
    ) -> None:
        super().__init__()

        self.checkpoint_activation = checkpoint_activation
        self.lam = lam
        # Define the window size of the neural network
        self.window_size = window_size
        assert all(
            [w_s % 2 == 0 for w_s in window_size]
        ), "Window size must be divisible by 2"
        self.shift_size = tuple([w_size // 2 for w_size in window_size])

        # Initialize serveral operations
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = LayerNorm(dim)
        self.pad3D = CustomPad3d(data_size[-3:], self.window_size)
        self.attention = EarthAttention3D(
            self.pad3D.padded_size, dim, num_heads, dropout_rate, self.window_size
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, dropout_rate=dropout_rate)

    def forward(self, x: Tensor, embedding_shape: torch.Size, roll: bool) -> Tensor:
        # Save the shortcut for skip-connection
        shortcut = x

        # Reshape input to three dimensions to calculate window attention
        # x = x.view((x.shape[0], Z, H, W, -1))
        x = x.view(embedding_shape)

        # Zero-pad input if needed
        # reshape data for padding, from B, Z, H, W, C to B, C, Z, H, W
        x = x.permute((0, 4, 1, 2, 3))
        x = self.pad3D(x)

        # back to previous shape
        x = x.permute((0, 2, 3, 4, 1))

        b, padded_z, padded_h, padded_w, c = x.shape

        if roll:
            # Roll x for half of the window for 3 dimensions
            x = x.roll(
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
            # Generate mask of attention masks
            # If two pixels are not adjacent, then mask the attention between them
            # Your can set the matrix element to -1000 when it is not adjacent,
            # then add it to the attention
            assert len(self.shift_size) == 3, "Shift size should be 3D"
            mask = generate_3d_attention_mask(
                x, self.window_size, self.shift_size, self.lam
            )
        else:
            # e.g., zero matrix when you add mask to attention
            mask = None

        # Reorganize data to calculate window attention
        x_window = x.reshape(
            shape=(
                x.shape[0],
                padded_z // self.window_size[0],
                self.window_size[0],
                padded_h // self.window_size[1],
                self.window_size[1],
                padded_w // self.window_size[2],
                self.window_size[2],
                -1,
            )
        )
        x_window = x_window.permute((0, 1, 3, 5, 2, 4, 6, 7))

        # Get data stacked in 3D cubes, which will further be used to
        # calculated attention among each cube
        x_window = x_window.reshape(
            shape=(
                -1,
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                c,
            )
        )

        # Apply 3D window attention with Earth-Specific bias
        if self.checkpoint_activation:
            x_window = checkpoint(
                self.attention,
                x_window,
                mask,
                b,
                padded_z,
                padded_h,
                use_reentrant=False,
            )
        else:
            x_window = self.attention(x_window, mask, b, padded_z, padded_h)

        # Reorganize data to original shapes
        x = x_window.reshape(
            shape=(
                b,
                padded_z // self.window_size[0],
                padded_h // self.window_size[1],
                padded_w // self.window_size[2],
                self.window_size[0],
                self.window_size[1],
                self.window_size[2],
                -1,
            )
        )
        x = x.permute((0, 1, 4, 2, 5, 3, 6, 7))

        # Reshape the tensor back to its original shape
        x = x.reshape(shape=(b, padded_z, padded_h, padded_w, -1))

        if roll:
            # Roll x back for half of the window
            x = x.roll(
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )

        # Crop the zero-padding
        (
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            padding_front,
            padding_back,
        ) = self.pad3D.padding
        x = x[
            :,
            padding_front : padded_z - padding_back,
            padding_top : padded_h - padding_bottom,
            padding_left : padded_w - padding_right,
            :,
        ]

        # Reshape the tensor back to the input shape
        x = x.reshape(shape=(b, -1, c))

        # Main calculation stages
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class EarthAttention3D(nn.Module):
    """3D sliding window attention with the Earth-Specific bias,
    see https://github.com/microsoft/Swin-Transformer for the official
    implementation of 2D sliding window attention.

    Args:
        data_size (torch.Size): data size in terms of plevel, latitude, longitude
        dim (int): token size
        num_heads (int): number of heads
        dropout_rate (float): dropout rate
        window_size (tuple[int]): window size (z, h ,w)
    """

    def __init__(
        self,
        data_size: torch.Size,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        window_size: tuple[int, int, int],
    ) -> None:
        super().__init__()

        # Store several attributes
        self.head_number = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        self.window_size = window_size

        # Construct position index to reuse self.earth_specific_bias
        self.register_buffer(
            "position_index", define_3d_earth_position_index(window_size)
        )

        # Init earth specific bias
        # only pressure level and latitude have absolute bias, longitude is cyclic
        # data_size is plevel, latitude, longitude
        self.num_windows = (data_size[0] // self.window_size[0]) * (
            data_size[1] // self.window_size[1]
        )

        # For each window, we will construct a set of parameters according to the paper
        # Inside a window, plevel and latitude positions are absolute while longitude are relative
        self.earth_specific_bias = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[2] - 1)
                * self.window_size[1] ** 2
                * self.window_size[0] ** 2,
                self.num_windows,
                self.head_number,
            )
        )

        # Initialize several operations
        self.linear1 = Linear(dim, dim * 3, bias=True)
        self.linear2 = Linear(dim, dim)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(p=dropout_rate)

        # Initialize the tensors using Truncated normal distribution
        torch.nn.init.trunc_normal_(self.earth_specific_bias, mean=0.0, std=0.02)

    def forward(self, x: Tensor, mask: Tensor, b: int, z: int, h: int) -> Tensor:
        # Record the original shape of the input (b*num_windows, window_size, dim)
        original_shape = x.shape

        # Linear layer to create query, key and value
        x = self.linear1(x)

        # reshape the data to calculate multi-head attention
        qkv = x.reshape(
            shape=(
                x.shape[0],
                x.shape[1],
                3,
                self.head_number,
                self.dim // self.head_number,
            )
        )
        query, key, value = qkv.permute((2, 0, 3, 1, 4))
        # 3, b*num_windows, head_number, window_size, dim_head

        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        self.attention = query @ key.mT
        # B*num_windows_lon*num_windows, head_number, window_size, window_size

        # self.earth_specific_bias is a set of neural network parameters to optimize.
        assert isinstance(self.position_index, Tensor)
        earth_specific_bias = self.earth_specific_bias[self.position_index]

        # Reshape the learnable bias to the same shape as the attention matrix
        earth_specific_bias = earth_specific_bias.reshape(
            shape=(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.num_windows,
                self.head_number,
            )
        )
        earth_specific_bias = earth_specific_bias.permute((2, 3, 0, 1))
        earth_specific_bias = earth_specific_bias.unsqueeze(0)
        # 1, num_windows, head_number, window_size, window_size

        # Add the Earth-Specific bias to the attention matrix
        attention_shape = self.attention.shape
        # Reshape and permute the lon dim to match the shape of earth_specific_bias
        attention = self.attention.reshape(
            b,
            self.num_windows,
            -1,
            self.head_number,
            attention_shape[-2],
            attention_shape[-1],
        )
        attention = attention.permute(0, 2, 1, 3, 4, 5)
        attention = attention.reshape(
            -1,
            self.num_windows,
            self.head_number,
            attention_shape[-2],
            attention_shape[-1],
        )
        # add bias
        attention = attention + earth_specific_bias
        # Reshape the attention matrix back
        attention = attention.reshape(
            b,
            -1,
            self.num_windows,
            self.head_number,
            attention_shape[-2],
            attention_shape[-1],
        )
        attention = attention.permute(0, 2, 1, 3, 4, 5)
        attention = attention.reshape(attention_shape)

        # Mask the attention between non-adjacent pixels, e.g., simply add
        # -100 to the masked element.
        if mask is not None:
            attention = attention.view(
                b, -1, self.head_number, original_shape[1], original_shape[1]
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(
                -1, self.head_number, original_shape[1], original_shape[1]
            )
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # Calculated the tensor after spatial mixing.
        x = attention @ value
        # B*num_windows, head_number, window_size, dim_head

        # Reshape tensor to the original shape
        x = x.permute((0, 2, 1, 3))  # B*num_windows, window_size, head_number, dim_head
        x = x.reshape(shape=original_shape)  # B*num_windows, window_size, dim

        # Linear layer to post-process operated tensor
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """MLP layers, same as most vision transformer architectures.

    Args:
        dim (int): input and output token size
        dropout_rate (float): dropout rate applied after each linear layer
    """

    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear1 = Linear(dim, dim * 4)
        self.linear2 = Linear(dim * 4, dim)
        self.activation = GELU()
        self.drop = Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x

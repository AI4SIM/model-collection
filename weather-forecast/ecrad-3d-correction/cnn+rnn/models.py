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


import os.path as osp
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import Conv1d, Linear, Dropout, Module, SiLU, Sequential
from torch.nn.functional import scaled_dot_product_attention, silu
from torch.nn.attention import SDPBackend, sdpa_kernel
from layers import HRLayer, MultiHeadAttention, PreProcessing

from typing import Union


class CNNModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        hidden_size: int = 512,
        kernel_size: int = 3,
        dilation_rates: Union[list[int], range] = [0, 1, 2, 3],
        conv_layers: int = 4,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attention_dropout: float = 0.0,
        flash_attention: bool = True,
        colum_padding: tuple[int, int, int, int] = (0, 0, 1, 0),
        inter_padding: tuple[int, int, int, int] = (0, 0, 1, 1),
        path_to_params: str = None,
    ):
        super().__init__()

        if isinstance(dilation_rates, range):
            dilation_rates = list(dilation_rates)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.conv_layers = conv_layers
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.flash_attention = flash_attention
        self.path_to_params = path_to_params
        self.colum_padding = colum_padding
        self.inter_padding = inter_padding

        # Preprocessing layer
        self.preprocess = PreProcessing(
            self.path_to_params, self.colum_padding, self.inter_padding
        )

        # Dilated convolution block to make the information propaate faster
        drates = [2**i for i in self.dilation_rates]
        layers = []
        for i, drate in enumerate(drates):
            if i == 0:
                layers.append(
                    Conv1d(
                        self.in_channels,
                        self.hidden_size,
                        kernel_size=self.kernel_size,
                        padding="same",
                        dilation=drate,
                    )
                )
            else:
                layers.append(
                    Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=self.kernel_size,
                        padding="same",
                        dilation=drate,
                    )
                )
            layers.append(SiLU())
        self.conv_block_with_dilation = Sequential(*layers)

        # Regular convolution block
        layers = []
        for _ in range(self.conv_layers):
            layers.append(
                Conv1d(
                    self.hidden_size,
                    self.hidden_size,
                    kernel_size=self.kernel_size,
                    padding="same",
                ),
            )
            layers.append(SiLU())
        self.conv_block = Sequential(*layers)

        # Attention layers
        self.mha_1 = MultiHeadAttention(
            self.hidden_size,
            self.num_heads,
            self.qkv_bias,
            self.attention_dropout,
            self.flash_attention,
        )
        self.mha_2 = MultiHeadAttention(
            self.hidden_size,
            self.num_heads,
            self.qkv_bias,
            self.attention_dropout,
            self.flash_attention,
        )

        # Flux layers
        # self.lw = Sequential(
        #     Conv1d(self.hidden_size, self.out_channels, 1, padding="same"),
        #     Linear(self.out_channels, self.out_channels, bias=True),
        # )
        self.lw_conv = Conv1d(self.hidden_size, self.out_channels, 1, padding="same")
        self.lw_lin = Linear(self.out_channels, self.out_channels, bias=True)

        self.sw_conv = Conv1d(self.hidden_size, self.out_channels, 1, padding="same")
        self.sw_lin = Linear(self.out_channels, self.out_channels, bias=True)
        # self.sw = Sequential(
        #     Conv1d(self.hidden_size, self.out_channels, 1, padding="same"),
        #     Linear(self.out_channels, self.out_channels, bias=True),
        # )
        # Heating rate layers
        self.hr_lw = HRLayer()
        self.hr_sw = HRLayer()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Preprocessing
        x = self.preprocess(inputs)

        # Convolutional layers
        B, T, C = x.size()  # batch size, sequence length, channels
        x = x.permute(0, 2, 1)  # B, C, T

        x = self.conv_block_with_dilation(x)
        x = x.permute(0, 2, 1)  # B, T, C
        x = self.mha_1(x)
        x = x.permute(0, 2, 1)  # B, C, T
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        x = self.mha_2(x)
        x = x.permute(0, 2, 1)

        # Flux layers
        lw = self.lw_conv(x)
        lw = lw.permute(0, 2, 1)
        lw = self.lw_lin(lw)  # B, T, 2

        sw = self.sw_conv(x)
        sw = sw.permute(0, 2, 1)
        sw = self.sw_lin(sw)  # B, T, 2

        # Heating rate layers
        hr_lw = self.hr_lw([lw, inputs["pressure_hl"]])
        hr_sw = self.hr_sw([sw, inputs["pressure_hl"]])

        return {
            "hr_sw": hr_sw,
            "hr_lw": hr_lw,
            "delta_sw_diff": sw[..., 0],
            "delta_sw_add": sw[..., 1],
            "delta_lw_diff": lw[..., 0],
            "delta_lw_add": lw[..., 1],
        }


# class RNNModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass

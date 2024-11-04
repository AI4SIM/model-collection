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
import pytorch_lightning as pl
from torch.nn import LazyConv1d, Linear, Dropout, Module, SiLU, Sequential
from torch.nn.functional import scaled_dot_product_attention, silu
from torch.nn.attention import SDPBackend, sdpa_kernel

from layers import HRLayer, MultiHeadAttention
from lightning import ThreeDCorrectionModule

from typing import Dict


class CNNModel(ThreeDCorrectionModule):
    def __init__(
        self,
        out_channels: int = 2,
        hidden_size: int = 512,
        kernel_size: int = 3,
        dilation_rates: list = [1, 2, 4, 8],
        conv_layers: int = 4,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attention_dropout: float = 0.0,
        flash_attention: bool = False,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.conv_layers = conv_layers
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.flash_attention = flash_attention

        self.save_hyperparameters()

        # Dilated convolution block to make the information propaate faster
        self.conv_block_with_dilation = Sequential()
        for drate in self.dilation_rates:
            self.conv_with_dilation.append(
                LazyConv1d(
                    self.hidden_size,
                    kernel_size=self.kernel_size,
                    padding="same",
                    dilation=drate,
                ),
                SiLU(),
            )
        # Regular convolution block
        self.conv_block = Sequential()
        for _ in range(self.conv_layers):
            self.conv.append(
                LazyConv1d(
                    self.hidden_size,
                    kernel_size=self.kernel_size,
                    padding="same",
                ),
                SiLU(),
            )
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
        self.lw = Sequential(LazyConv1d(2, 1, padding="same"), Linear(2, 2, bias=True))
        self.sw = Sequential(LazyConv1d(2, 1, padding="same"), Linear(2, 2, bias=True))
        # Heating rate layers
        self.hr_lw = HRLayer()
        self.hr_sw = HRLayer()

    def forward(
        self, x: torch.Tensor, pressure_hl: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.size()  # batch size, sequence length, channels

        x = self.conv_block_with_dilation(x)
        x = self.mha_1(x)
        x = self.conv_block(x)
        x = self.mha_2(x)

        # Flux layers
        lw = self.lw_conv(x)  # B, T, 2
        sw = self.sw_conv(x)  # B, T, 2

        # Heating rate layers
        hr_lw = self.hr_lw([lw, pressure_hl])
        hr_sw = self.hr_sw([sw, pressure_hl])

        return {
            "hr_sw": hr_sw,
            "hr_lw": hr_lw,
            "delta_sw_diff": sw[..., 0],
            "delta_sw_add": sw[..., 1],
            "delta_lw_diff": lw[..., 0],
            "delta_lw_add": lw[..., 1],
        }


class RNNModel(ThreeDCorrectionModule):
    raise ("Not implemented yet")

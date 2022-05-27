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

from torch import cat
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    DoubleConv block (kernel_size conv -> BN -> ReLU) ** 2.
    The residual option allows to short-circuit the convolutions
    with an additive residual (ResNet-like).
    """

    def __init__(self,
                 inp_ch: int,
                 out_ch: int,
                 residual: bool = False,
                 kernel_size: int = 3,
                 dim: int = 3):
        super().__init__()

        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}.get(dim)
        batchnorm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}.get(dim)

        self.net = nn.Sequential(
            conv(inp_ch, out_ch, kernel_size=kernel_size, padding=1),
            batchnorm(out_ch),
            nn.ReLU(inplace=True),
            conv(out_ch, out_ch, kernel_size=kernel_size, padding=1),
            batchnorm(out_ch),
            nn.ReLU(inplace=True))

        self.add_res = None
        if residual:
            self.add_res = conv(inp_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.net(x) + (self.add_res(x) if self.add_res is not None else 0)


class Downsampler(nn.Module):
    """Combination of MaxPool1d and DoubleConv in series."""

    def __init__(self,
                 inp_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 dim: int = 3):
        super().__init__()

        maxpool = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}.get(dim)

        self.net = nn.Sequential(
            maxpool(kernel_size=2, stride=2),
            DoubleConv(inp_ch, out_ch, kernel_size=kernel_size, dim=dim))

    def forward(self, x):
        return self.net(x)


class Upsampler(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double convolution.
    """

    def __init__(self,
                 inp_ch: int,
                 out_ch: int,
                 bilinear: bool = False,
                 kernel_size: int = 3,
                 dim: int = 3):
        super().__init__()
        self.upsample = None
        self.dim = dim

        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}.get(dim)
        conv_t = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}.get(dim)

        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                conv(inp_ch, inp_ch // 2, kernel_size=1))
        else:
            self.upsample = conv_t(inp_ch, inp_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(inp_ch, out_ch, kernel_size=kernel_size, dim=dim)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2.
        d2 = x2.shape[2] - x1.shape[2]
        pad = [d2 // 2, d2 - d2 // 2]
        if self.dim > 1:
            d3 = x2.shape[3] - x1.shape[3]
            pad.extend([d3 // 2, d3 - d3 // 2])
        if self.dim > 2:
            d4 = x2.shape[4] - x1.shape[4]
            pad.extend([d4 // 2, d4 - d4 // 2])
        pad.reverse()  # pads from last to first.
        x1 = nn.functional.pad(x1, pad)

        x = cat([x2, x1], dim=1)  # concatenate along the channels axis.
        return self.conv(x)

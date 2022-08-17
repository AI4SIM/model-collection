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

from torch import cat, Tensor
import torch.nn as nn


class UNet3D(nn.Module):
    """
    U-net with "fully" 3D convolutions (assuming isotropicity, learning 3D kernels).
    Using float64 (double-precision) for physics.
    Args:
        inp_ch: Number of channels of the input.
        out_ch: Number of channels of the output.
        n_levels: Number of levels (up/down-sampler + double conv).
        n_features_root: Number of features in the first level, squared at each level.
        bilinear: Whether to use bilinear interpolation or transposed convolutions for upsampling.
    """

    def __init__(self,
                 inp_ch: int,
                 out_ch: int,
                 n_levels: int,
                 n_features_root: int,
                 bilinear: bool = False):
        super().__init__()
        self.n_levels = n_levels

        # First level hardcoded.
        layers = [DoubleConv(inp_ch, n_features_root)]

        # Downward path.
        f = n_features_root
        for _ in range(n_levels - 1):
            layers.append(Downsampler(f, f * 2))
            f *= 2

        # Upward path.
        for _ in range(n_levels - 1):
            layers.append(Upsampler(f, f // 2, bilinear))
            f //= 2

        layers.append(DoubleConv(f, out_ch))
        self.layers = nn.ModuleList(layers).double()  # forces double precision for the whole model.

    def forward(self, x: Tensor) -> Tensor:
        # xi keeps the data at each level, allowing to pass it through skip-connections.
        xi = [self.layers[0](x)]

        # Downward path.
        for layer in self.layers[1:self.n_levels]:
            xi.append(layer(xi[-1]))

        # Upward path.
        for i, layer in enumerate(self.layers[self.n_levels:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])  # upsamplers taking skip-connections.

        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """
    DoubleConv block (3x3x3 conv -> BN -> ReLU) ** 2.
    The residual option allows to short-circuit the convolutions
    with an additive residual (ResNet-like).
    """

    def __init__(self, inp_ch: int, out_ch: int, residual: bool = False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(inp_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

        self.res = None
        if residual:
            self.res = nn.Conv3d(inp_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + (self.res(x) if self.res is not None else 0)


class Downsampler(nn.Module):
    """Combination of MaxPool3d and DoubleConv in series."""

    def __init__(self, inp_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(inp_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Upsampler(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double 3x3x3 convolution.
    """

    def __init__(self, inp_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if inp_ch < 2:
            raise ValueError('Input channel ({}) too low.'.format(inp_ch))
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv3d(inp_ch, inp_ch // 2, kernel_size=1))
        else:
            self.upsample = nn.ConvTranspose3d(inp_ch, inp_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(inp_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2.
        d2 = x2.shape[2] - x1.shape[2]
        d3 = x2.shape[3] - x1.shape[3]
        d4 = x2.shape[4] - x1.shape[4]
        pad = [d4 // 2, d4 - d4 // 2,  # from last to first.
            d3 // 2, d3 - d3 // 2,
            d2 // 2, d2 - d2 // 2]
        x1 = nn.functional.pad(x1, pad)

        x = cat([x2, x1], dim=1)  # concatenate along the channels axis.
        return self.conv(x)

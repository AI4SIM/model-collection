'''
    Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    *     http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
'''

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    """
    U-net with "fully" 3D convolutions (assuming isotropicity, learning 3D kernels).
    Using float64 (double-precision) for physics.

    Args:
        inp_feat: Number of channels of the input.
        out_feat: Number of channels of the output.
        n_levels: Number of levels (up/down-sampler + double conv).
        n_features_root: Number of features in the first level, squared at each level.
        bilinear: Whether to use bilinear interpolation or transposed convolutions for upsampling.
    """

    def __init__(self, inp_feat: int, out_feat: int, n_levels: int, n_features_root: int, bilinear: bool = False):
        super().__init__()
        self.n_levels = n_levels

        # First level hardcoded.
        layers = [DoubleConv(inp_feat, n_features_root)]

        # Downward path.
        f = n_features_root
        for _ in range(n_levels - 1):
            layers.append(Downsampler(f, f * 2))
            f *= 2

        # Upward path.
        for _ in range(n_levels - 1):
            layers.append(Upsampler(f, f // 2, bilinear))
            f //= 2

        layers.append(DoubleConv(f, out_feat))
        self.layers = nn.ModuleList(layers).double()

    def forward(self, x):
        # xi keeps the data at each level, allowing to pass it through skip-connections.
        xi = [self.layers[0](x)]
        for layer in self.layers[1:self.n_levels]:  # downward path.
            xi.append(layer(xi[-1]))
        for i, layer in enumerate(self.layers[self.n_levels:-1]):  # upward path.
            xi[-1] = layer(xi[-1], xi[-2 - i])  # upsamplers taking skip-connections.
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """
    DoubleConv block (3x3x3 conv -> BN -> ReLU) ** 2.
    The residual option allows to short-circuit the convolutions with an additive residual (ResNet-like).
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
        if residual: self.additive_residual = nn.Conv3d(inp_ch, out_ch, kernel_size=1, bias=False)
        else: self.additive_residual = None

    def forward(self, x):
        if self.additive_residual is not None: return self.net(x) + self.additive_residual(x)
        else: return self.net(x)


class Downsampler(nn.Module):
    """
    Combination of MaxPool3d and DoubleConv in series.
    """

    def __init__(self, inp_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), DoubleConv(inp_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Upsampler(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by double 3x3x3 convolution.
    """

    def __init__(self, inp_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv3d(inp_ch, inp_ch // 2, kernel_size=1))
        else:
            self.upsample = nn.ConvTranspose3d(inp_ch, inp_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(inp_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2.
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1 = torch.nn.functional.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

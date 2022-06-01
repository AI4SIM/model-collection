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

import torch.nn as nn

from samplers import DoubleConv, Upsampler, Downsampler


class UNet(nn.Module):
    """
    Implements a U-net (parameterizable dimension).
    Using float64 (double-precision) for physics.

    Args:
        dim (int): Number of dimensions of the data (e.g. 1, 2 or 3).
        inp_ch (int): Number of channels of the input.
        out_ch (int): Number of channels of the output.
        n_levels (int): Number of levels (up/down-sampler + double conv).
        n_features_root (int): Number of features in the first level, squared at each level.
        bilinear (bool): Whether to use bilinear interpolation or transposed convolutions for upsampling.
    """

    def __init__(self,
                 dim: int,
                 inp_ch: int,
                 out_ch: int,
                 n_levels: int = 3,
                 n_features_root: int = 32,
                 bilinear: bool = False):
        super().__init__()
        self.n_levels = n_levels

        # First level hardcoded.
        layers = [DoubleConv(dim, inp_ch, n_features_root)]

        # Downward path.
        f = n_features_root
        for _ in range(n_levels - 1):
            layers.append(Downsampler(dim, f, f * 2))
            f *= 2

        # Upward path.
        for _ in range(n_levels - 1):
            layers.append(Upsampler(dim, f, f // 2, bilinear))
            f //= 2

        layers.append(DoubleConv(dim, f, out_ch))
        self.layers = nn.ModuleList(layers).double()

    def forward(self, x):
        # xi keeps the data at each level, allowing to pass it through skip-connections.
        xi = [self.layers[0](x)]

        # Downward path.
        for layer in self.layers[1:self.n_levels]:
            xi.append(layer(xi[-1]))

        # Upward path.
        for i, layer in enumerate(self.layers[self.n_levels:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])  # upsamplers taking skip-connections.

        return self.layers[-1](xi[-1])

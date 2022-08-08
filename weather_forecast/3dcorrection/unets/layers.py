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

import torch
from torch.nn import Module


class HRLayer(Module):
    """
    Layer to calculate heating rates given fluxes and half-level pressures.
    This could be used to deduce the heating rates within the model so that
    the outputs can be constrained by both fluxes and heating rates.
    """
    def __init__(self):
        super().__init__()
        self.g_cp = torch.tensor(24 * 3600 * 9.80665 / 1004)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        hlpress = inputs[1]
        net_press = hlpress[..., 1:] - hlpress[..., :-1]
        for i in [0, 2]:
            netflux = inputs[0][..., i]
            flux_diff = netflux[..., 1:] - netflux[..., :-1]
            outputs.append(-self.g_cp * torch.Tensor.divide(flux_diff, net_press))
        return torch.stack(outputs, dim=-1)
    

class Normalization(Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = torch.tensor(1.e-12)
        self.register_buffer("feat_mean", self.mean)
        self.register_buffer("feat_std", self.std)
        self.register_buffer("epsilon", self.eps)

    def forward(self, x):
        self.mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        self.std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        self.eps = torch.as_tensor(self.eps, dtype=x.dtype, device=x.device)
        assert x.shape[1:] == self.mean.shape
        assert x.shape[1:] == self.std.shape
        return x.sub(self.mean).div(self.std + self.eps)
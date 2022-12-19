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
from typing import Dict
from torch.nn import Module, ZeroPad2d

icol_keys = [
    "q",
    "o3_mmr",
    "co2_vmr",
    "n2o_vmr",
    "ch4_vmr",
    "o2_vmr",
    "cfc11_vmr",
    "cfc12_vmr",
    "hcfc22_vmr",
    "ccl4_vmr",
    "cloud_fraction",
    "aerosol_mmr",
    "q_liquid",
    "q_ice",
    "re_liquid",
    "re_ice",
]
ihl_keys = ["temperature_hl", "pressure_hl"]
iinter_keys = ["overlap_param"]
isca_keys = [
    "skin_temperature",
    "cos_solar_zenith_angle",
    "sw_albedo",
    "sw_albedo_direct",
    "lw_emissivity",
    "solar_irradiance",
]


# TO TEST
class HRLayer(Module):
    """
    Layer to calculate heating rates given fluxes and half-level pressures.
    This could be used to deduce the heating rates within the model so that
    the outputs can be constrained by both fluxes and heating rates.
    """
    def __init__(self) -> None:
        super().__init__()
        self.g_cp = torch.tensor(24 * 3600 * 9.80665 / 1004)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hlpress = inputs[1]
        net_press = hlpress[..., 1:] - hlpress[..., :-1]
        netflux = inputs[0][..., 0]
        flux_diff = netflux[..., 1:] - netflux[..., :-1]

        return -self.g_cp * torch.Tensor.divide(flux_diff, net_press)

class Normalization(Module):
    """
    Layer to normalize the data per batch.
    Requires the means and standard deviations of the features.
    They are then saved as registered buffers inside the model.
    """
    def __init__(self, mean: torch.Tensor = None, std: torch.Tensor = None, label: str = "") -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = torch.tensor(1.e-12)
        self.register_buffer(label + "mean", self.mean)
        self.register_buffer(label + "std", self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass."""
        # self.mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        # self.std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        # # self.eps = torch.as_tensor(self.eps, dtype=x.dtype, device=x.device)
        # assert x.shape[1:] == self.mean.shape[1:]
        # assert x.shape[1:] == self.std.shape[1:]
        #
        # return x.sub(self.mean).div(self.std + self.eps)
        return x

class Layer(Module):
    """
    Base class for preprocessing the inputs. Normalization is instanciated here.
    """
    def __init__(self, mean: dict, std: dict, label: str) -> None:
        super().__init__()
        
        self.mean = mean
        self.std = std
        self.normalization = Normalization(self.mean, self.std, label=label)

    def forward(self, x):
        raise NotImplementedError

class ScaLayer(Layer):
    """Merge the scalar inputs."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__(mean, std, label="sca_")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.
        
        Args:
            x: Dictionary containing all the features.
            
        Returns:
            Resulting model forward pass.
        """
        tmp = []
        for key in isca_keys:
            if key in ["skin_temperature", "cos_solar_zenith_angle", "solar_irradiance"]:
                inputs = x[key].unsqueeze(dim=-1)
            else:
                inputs = x[key]
            tmp.append(inputs)                

        inputs = torch.cat(tmp, dim=-1)
        inputs = self.normalization(inputs)

        return inputs

class ColLayer(Layer):
    """Merge the column inputs (given on 137 vertical levels)."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__(mean, std, label="col_")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.
        
        Args:
            x: Dictionary containing all the features.
            
        Returns:
            Resulting model forward pass.
        """
        tmp = []
        for key in icol_keys:
            if key == "aerosol_mmr":
                inputs = x[key].permute((0, 2, 1))
            else:
                inputs = x[key].unsqueeze(dim=-1)
            tmp.append(inputs)
        
        inputs = torch.cat(tmp, dim=-1)
        inputs = self.normalization(inputs)

        return inputs

class HLLayer(Layer):
    """Merge the half-level inputs (given on 138 vertical half-levels)."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__(mean, std, label="hl_")

    def forward(self, x):
        """Compute the forward pass.
        
        Args:
            x: Dictionary containing all the features.
            
        Returns:
            Resulting model forward pass.
        """
        temperature_hl = x["temperature_hl"].unsqueeze(dim=-1)
        pressure_hl = x["pressure_hl"].unsqueeze(dim=-1)
        inputs = torch.cat([temperature_hl, pressure_hl], dim=-1)
        inputs = self.normalization(inputs)
        
        return inputs

class InterLayer(Layer):
    """Normalize the interface input (given on 136 levels)."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__(mean, std, label="inter_")

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.
        
        Args:
            x: Dictionary containing all the features.
            
        Returns:
            Resulting model forward pass.
        """
        inputs = x["overlap_param"].unsqueeze(dim=-1)
        inputs = self.normalization(inputs)
        
        return inputs

class PreProcessing(Module):
    """
    Layer to preprocess the feature tensors:
        * Reshape and concatenate the features (scalar, column, half-level and inter inputs),
        * Normalize them
        * Repeat Vector, Pad and Concatenate
    """
    def __init__(self, means: Dict[str, torch.Tensor], stds: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        
        self.means = means
        self.stds = stds
        
        self.sca_layer = ScaLayer(self.means["sca_inputs"], self.stds["sca_inputs"])
        self.col_layer = ColLayer(self.means["col_inputs"], self.stds["col_inputs"])
        self.hl_layer = HLLayer(self.means["hl_inputs"], self.stds["hl_inputs"])
        self.inter_layer = InterLayer(self.means["inter_inputs"], self.stds["inter_inputs"])
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.
        
        Args:
            x: Dictionary containing all the features.
            
        Returns:
            Resulting model forward pass.
        """
        sca_inputs = self.sca_layer(x)
        col_inputs = self.col_layer(x)
        hl_inputs = self.hl_layer(x)
        inter_inputs = self.inter_layer(x)
        
        # Stack properly all the inputs
        sca_in = sca_inputs.unsqueeze(dim=-1).expand(*sca_inputs.size(), 138).permute(0, 2, 1)
        col_in = ZeroPad2d((0, 0, 1, 0))(col_inputs)
        inter_in = ZeroPad2d((0, 0, 1, 1))(inter_inputs)
        
        return torch.cat((sca_in, col_in, hl_inputs, inter_in), dim=-1)
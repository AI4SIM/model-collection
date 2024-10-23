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

from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Keys:
    """
    Dataclass to store the keys of the dictionary that contains the input data.
    """
    # Scalar variables
    isca_keys: Tuple[str, ...] = (
        "skin_temperature",
        "cos_solar_zenith_angle",
        "sw_albedo",
        "sw_albedo_direct",
        "lw_emissivity",
        "solar_irradiance",
    )

    # 137-levels variables
    icol_keys: Tuple[str, ...] = (
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
    )

    # 138-levels variables
    ihl_keys: Tuple[str, ...] = (
        "temperature_hl",
        "pressure_hl",
    )

    # 136-levels variables
    iinter_keys: Tuple[str, ...] = (
        "overlap_param"
    )
    
    
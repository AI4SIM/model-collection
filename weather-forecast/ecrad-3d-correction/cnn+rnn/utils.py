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

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class Keys:
    """
    Dataclass to store the keys of the dictionary that contains the input data.
    """

    # Scalar variables
    isca_keys: tuple[str, ...] = (
        "skin_temperature",
        "cos_solar_zenith_angle",
        "sw_albedo",
        "sw_albedo_direct",
        "lw_emissivity",
        "solar_irradiance",
    )

    # 137-levels variables
    icol_keys: tuple[str, ...] = (
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
    ihl_keys: tuple[str, ...] = (
        "temperature_hl",
        "pressure_hl",
    )

    # 136-levels variables
    iinter_keys: tuple[str, ...] = ("overlap_param",)

    # Input variables
    input_keys: tuple[str, ...] = isca_keys + icol_keys + ihl_keys + iinter_keys

    # Output variables
    output_keys: tuple[str, ...] = ("sw", "lw", "hr_sw", "hr_lw")


@dataclass(frozen=True)
class VarInfo:
    # Scalar variable info
    sca_variables: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "skin_temperature": {"shape": [], "idx": 0},
            "cos_solar_zenith_angle": {"shape": [1], "idx": 1},
            "sw_albedo": {"shape": [6], "idx": range(2, 2 + 6)},
            "sw_albedo_direct": {"shape": [6], "idx": range(8, 8 + 6)},
            "lw_emissivity": {"shape": [2], "idx": range(14, 14 + 2)},
            "solar_irradiance": {"shape": [], "idx": -1},
        }
    )
    # Column variable info
    col_variables: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "q": {"shape": [137], "idx": 0},
            "o3_mmr": {"shape": [137], "idx": 1},
            "co2_vmr": {"shape": [137], "idx": 2},
            "n2o_vmr": {"shape": [137], "idx": 3},
            "ch4_vmr": {"shape": [137], "idx": 4},
            "o2_vmr": {"shape": [137], "idx": 5},
            "cfc11_vmr": {"shape": [137], "idx": 6},
            "cfc12_vmr": {"shape": [137], "idx": 7},
            "hcfc22_vmr": {"shape": [137], "idx": 8},
            "ccl4_vmr": {"shape": [137], "idx": 9},
            "cloud_fraction": {"shape": [137], "idx": 10},
            "aerosol_mmr": {"shape": [137, 12], "idx": range(11, 11 + 12)},
            "q_liquid": {"shape": [137], "idx": 3},
            "q_ice": {"shape": [137], "idx": 24},
            "re_liquid": {"shape": [137], "idx": 25},
            "re_ice": {"shape": [137], "idx": 26},
        }
    )
    # Half-level variable info
    hl_variables: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "temperature_hl": {"shape": [138], "idx": 0},
            "pressure_hl": {"shape": [138], "idx": 1},
        }
    )
    # Inter-level variable info
    inter_variables: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "overlap_param": {"shape": [136], "idx": 0},
        }
    )

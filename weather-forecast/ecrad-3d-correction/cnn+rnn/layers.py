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

from utils import Keys, get_means_and_stds

import torch
from torch.nn import Dropout, Linear, Module, ZeroPad2d
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


class MultiHeadAttention(Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool,
        attention_dropout: float,
        flash_attention: bool,
    ) -> None:
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "Hidden size must be divisible by the number of heads."

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.flash_attention = flash_attention

        self.qkv = Linear(self.hidden_size, self.hidden_size * 3, bias=self.qkv_bias)
        self.proj = Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.resid_dropout = Dropout(p=self.attention_dropout)

        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        k = k.view(B, T, self.num_heads, self.hidden_size // self.num_heads).transpose(
            1, 2
        )  # (B, nH, T, hs)
        q = q.view(B, T, self.num_heads, self.hidden_size // self.num_heads).transpose(
            1, 2
        )  # (B, nH, T, hs)
        v = v.view(B, T, self.num_heads, self.hidden_size // self.num_heads).transpose(
            1, 2
        )  # (B, nH, T, hs)

        # Scaled dot-product attention
        if self.flash_attention:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False,
                )
        else:
            with sdpa_kernel(SDPBackend.MATH):
                x = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False,
                )
        x = x.transpose(1, 2).contiguous().view(B, T, self.hidden_size)

        # Output projection
        x = self.resid_dropout(self.proj(x))

        return x

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.qkv.weight)
        torch.nn.init.xavier_uniform_(self.proj.weight)

        if self.qkv_bias:
            torch.nn.init.zeros_(self.qkv.bias)
            torch.nn.init.zeros_(self.proj.bias)


class HRLayer(Module):
    """
    Layer to calculate heating rates given fluxes and half-level pressures.
    This could be used to deduce the heating rates within the model so that
    the outputs can be constrained by both fluxes and heating rates.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("g_cp", torch.tensor(24 * 3600 * 9.80665 / 1004))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # test the input shape
        netflux = inputs[0][..., 0]
        hlpress = inputs[1]
        net_press = hlpress[..., 1:] - hlpress[..., :-1]
        flux_diff = netflux[..., 1:] - netflux[..., :-1]

        return -self.g_cp * torch.Tensor.divide(flux_diff, net_press)


class Normalization(Module):
    """
    Layer to normalize the data per batch.
    Requires the means and standard deviations of the features.
    They are then saved as registered buffers inside the model.

    Args:
        mean (torch.Tensor): Mean values of the features.
        std (torch.Tensor): Standard deviation values of features.

    Returns:
        torch.Tensor:
            Normalized data.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()

        self.mean = mean
        self.std = std
        self.register_buffer("eps", torch.tensor(1.0e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass."""
        assert x.shape[1:] == self.mean.shape
        assert x.shape[1:] == self.std.shape

        return x.sub(self.mean).div(self.std + self.eps)


# class Layer(Module):
#     """
#     Base class for preprocessing the inputs. Normalization is instanciated here.
#     """
#     def __init__(self, mean: dict, std: dict, label: str) -> None:
#         super().__init__()

#         self.mean = mean
#         self.std = std
#         self.normalization = Normalization(self.mean, self.std, label=label)

#     def forward(self, x):
#         raise NotImplementedError


class ScaLayer(Module):
    """
    Merge the scalar inputs.

    Args:
        mean (torch.Tensor): Mean values of the scalar features.
        std (torch.Tensor): Standard deviation values of the scalar features.

    Returns:
        torch.Tensor:
            All 'scalar' variables concatenated and normalized.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()

        self.register_buffer("sca_mean", mean)
        self.register_buffer("sca_std", std)

        self.normalization = Normalization(self.sca_mean, self.sca_std)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            x (torch.Tensor): dictionary containing all the features.

        Returns:
            torch.Tensor:
                Resulting model forward pass.
        """
        scalar_variables = []
        for key in Keys().isca_keys:
            if key in [
                "skin_temperature",
                "cos_solar_zenith_angle",
                "solar_irradiance",
            ]:
                inputs = x[key].unsqueeze(dim=-1)
            else:
                inputs = x[key]
            scalar_variables.append(inputs)

        inputs = torch.cat(scalar_variables, dim=-1)
        inputs = self.normalization(inputs)

        return inputs


class ColLayer(Module):
    """
    Merge the column inputs (given on 137 vertical levels).

    Args:
        mean (torch.Tensor): Mean values of the column features.
        std (torch.Tensor): Standard deviation values of the column features.

    Returns:
        torch.Tensor:
            All 'column' variables concatenated and normalized.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()

        self.register_buffer("col_mean", mean)
        self.register_buffer("col_std", std)

        self.normalization = Normalization(self.col_mean, self.col_std)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            x (torch.Tensor): dictionary containing all the features.

        Returns:
            torch.Tensor:
                Resulting model forward pass.
        """
        column_variables = []
        for key in Keys().icol_keys:
            if key == "aerosol_mmr":
                inputs = x[key].permute((0, 2, 1))
            else:
                inputs = x[key].unsqueeze(dim=-1)
            column_variables.append(inputs)

        inputs = torch.cat(column_variables, dim=-1)
        inputs = self.normalization(inputs)

        return inputs


class HLLayer(Module):
    """
    Merge the half-level inputs (given on 138 vertical half-levels).

    Args:
        mean (torch.Tensor): Mean values of the half-level features.
        std (torch.Tensor): Standard deviation values of the half-level features.

    Returns:
        torch.Tensor:
            All 'half-level' variables concatenated and normalized.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()

        self.register_buffer("hl_mean", mean)
        self.register_buffer("hl_std", std)

        self.normalization = Normalization(self.hl_mean, self.hl_std)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            x (torch.Tensor): dictionary containing all the features.

        Returns:
            torch.Tensor:
                Resulting model forward pass.
        """
        hl_variables = []
        for key in Keys().ihl_keys:
            inputs = x[key].unsqueeze(dim=-1)
            hl_variables.append(inputs)

        inputs = torch.cat(hl_variables, dim=-1)
        inputs = self.normalization(inputs)

        return inputs


class InterLayer(Module):
    """
    Normalize the interface input (given on 136 levels).

    Args:
        mean (torch.Tensor): Mean values of overlap_param.
        std (torch.Tensor): Standard deviation values of overlap_param.

    Returns:
        torch.Tensor:
            overlap_param normalized.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()

        self.register_buffer("inter_mean", mean)
        self.register_buffer("inter_std", std)

        self.normalization = Normalization(self.inter_mean, self.inter_std)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            x (torch.Tensor): dictionary containing all the features.

        Returns:
            torch.Tensor:
                Resulting model forward pass.
        """
        inputs = x[Keys().iinter_keys[0]].unsqueeze(dim=-1)
        inputs = self.normalization(inputs)

        return inputs


class PreProcessing(Module):
    """
    Layer to preprocess the feature tensors:
        * Reshape and concatenate the features (scalar, column, half-level and inter inputs),
        * Normalize them
        * Repeat Vector, Pad and Concatenate
    """

    def __init__(
        self,
        path_to_params: str = None,
        colum_padding: tuple[int, int, int, int] = (0, 0, 1, 0),
        inter_padding: tuple[int, int, int, int] = (0, 0, 1, 1),
    ) -> None:
        super().__init__()

        self.means, self.stds = get_means_and_stds(path_to_params)

        self.sca_layer = ScaLayer(self.means["sca_inputs"], self.stds["sca_inputs"])
        self.col_layer = ColLayer(self.means["col_inputs"], self.stds["col_inputs"])
        self.hl_layer = HLLayer(self.means["hl_inputs"], self.stds["hl_inputs"])
        self.inter_layer = InterLayer(
            self.means["inter_inputs"], self.stds["inter_inputs"]
        )

        self.zeropad_col = ZeroPad2d(colum_padding)
        self.zeropad_inter = ZeroPad2d(inter_padding)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            x (torch.Tensor): dictionary containing all the features.

        Returns:
            torch.Tensor:
                Resulting model forward pass.
        """
        sca_inputs = self.sca_layer(x)
        col_inputs = self.col_layer(x)
        hl_inputs = self.hl_layer(x)
        inter_inputs = self.inter_layer(x)

        # Stack properly all the inputs
        sca_in = (
            sca_inputs.unsqueeze(dim=-1)
            .expand(sca_inputs.size(dim=0), sca_inputs.size(dim=1), 138)
            .permute(0, 2, 1)
        )
        col_in = self.zeropad_col(col_inputs)
        inter_in = self.zeropad_inter(inter_inputs)

        dim_to_check = 1
        assert (
            sca_in.size(dim=dim_to_check)
            == col_in.size(dim=dim_to_check)
            == hl_inputs.size(dim=dim_to_check)
            == inter_in.size(dim=dim_to_check)
        ), f"Dimension mismatch: {sca_in.size(dim=dim_to_check)} != {col_in.size(dim=dim_to_check)} != {hl_inputs.size(dim=dim_to_check)} != {inter_in.size(dim=dim_to_check)}"

        return torch.cat((sca_in, col_in, hl_inputs, inter_in), dim=-1)

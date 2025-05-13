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
from torch import Tensor


def check_same_coordinates(
    longitude_1: Tensor, latitude_1: Tensor, longitude_2: Tensor, latitude_2: Tensor
) -> bool:
    """Check that 2 coordinate systems match, i. e., the longitudes and latitudes are the same

    Args:
        longitude_1 (Tensor): reference longitude
        latitude_1 (Tensor): reference latitude
        longitude_2 (Tensor): longitude to check
        latitude_2 (Tensor): latitude to check

    Returns:
        bool
    """
    longitude_match = torch.all(longitude_1 == longitude_2)
    latitude_match = torch.all(latitude_1, latitude_2)
    return longitude_match and latitude_match


def cat_constant_masks(surface_data: Tensor, constant_masks: Tensor) -> Tensor:
    """Concatenate constant masks to surface data.

    Args:
        surface_data (Tensor): surface data
        constant_masks (Tensor): Tensor of masks

    Returns:
        Tensor: constant masks concatenated with surface data
    """
    batch_size = surface_data.shape[0]
    mask_data = constant_masks.expand(batch_size, *constant_masks.shape)
    surface_data = torch.cat([surface_data, mask_data], dim=1)
    return surface_data


def generate_3d_attention_mask(
    x: Tensor, window_size: tuple[int], shift_size: tuple[int], lam: bool = False
) -> Tensor:
    """Generate attention mask for sliding window attention in the context of 3D data.
    Based on:
    https://pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html#swin_s

    Args:
        x (Tensor): input data, used to generate the mask on the same device
        window_size (tuple[int]): size of the sliding window
        shift_size (tuple[int]): size of the shift for the sliding window

    Returns:
        Tensor: attention mask
    """
    assert x.dim() == 5, f"Data must be 3D, but has {x.dim()} dimension(s)"
    _, pad_z, pad_h, pad_w, _ = x.shape
    assert (
        pad_z % window_size[0] == 0
        and pad_h % window_size[1] == 0
        and pad_w % window_size[2] == 0
    ), "the data size must divisible by window size"
    assert (
        window_size[0] % shift_size[0] == 0
        and window_size[1] % shift_size[1] == 0
        and window_size[2] % shift_size[2] == 0
    ), "the window size must divisible by shift size"

    # Create the attention mask from the data to have same type and device
    attention_mask = x.new_zeros((pad_z, pad_h, pad_w))
    z_slices = ((0, -shift_size[0]), (-shift_size[0], None))
    h_slices = ((0, -shift_size[1]), (-shift_size[1], None))

    if lam:
        w_slices = ((0, -shift_size[2]), (-shift_size[2], None))
    else:
        w_slices = ((0, None),)

    count = 0
    for z in z_slices:
        for h in h_slices:
            for w in w_slices:
                attention_mask[z[0] : z[1], h[0] : h[1], w[0] : w[1]] = count
                count += 1

    attention_mask = attention_mask.reshape(
        pad_z // window_size[0],
        window_size[0],
        pad_h // window_size[1],
        window_size[1],
        pad_w // window_size[2],
        window_size[2],
    )
    num_windows = (
        (pad_z // window_size[0])
        * (pad_h // window_size[1])
        * (pad_w // window_size[2])
    )
    attention_mask = torch.permute(attention_mask, (0, 2, 4, 1, 3, 5)).reshape(
        num_windows, -1
    )
    attention_mask = attention_mask.unsqueeze(1) - attention_mask.unsqueeze(2)
    attention_mask.masked_fill_(attention_mask != 0, -100.0)
    return attention_mask


def generate_2d_swin_masks(
    pad_h: int,
    pad_w: int,
    window_size: tuple[int],
    shift_size: tuple[int],
    num_windows: int,
) -> Tensor:
    """Original method to generate attention mask for sliding window attention of Swin transformer.
    See https://pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html#swin_s

    Args:
        pad_h (int): height size of the data after padding
        pad_w (int): width size of the data after padding
        window_size (tuple[int]): size of the sliding window
        shift_size (tuple[int]): size of the shift for the sliding window
        num_windows (int): number of windows

    Returns:
        Tensor: attention mask
    """
    # generate attention mask
    attn_mask = torch.zeros((pad_h, pad_w))
    h_slices = (
        (0, -window_size[0]),
        (-window_size[0], -shift_size[0]),
        (-shift_size[0], None),
    )
    w_slices = (
        (0, -window_size[1]),
        (-window_size[1], -shift_size[1]),
        (-shift_size[1], None),
    )
    count = 0
    for h in h_slices:
        for w in w_slices:
            attn_mask[h[0] : h[1], w[0] : w[1]] = count
            count += 1
    attn_mask = attn_mask.view(
        pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1]
    )
    attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
        num_windows, window_size[0] * window_size[1]
    )
    attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


def generate_modified_2d_swin_masks(
    pad_h: int,
    pad_w: int,
    window_size: tuple[int],
    shift_size: tuple[int],
    num_windows: int,
) -> Tensor:
    """Generate attention mask for sliding window attention of Swin transformer.
    Based on
    https://pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html#swin_s,
    but simplified

    Args:
        pad_h (int): height size of the data after padding
        pad_w (int): width size of the data after padding
        window_size (tuple[int]): size of the sliding window
        shift_size (tuple[int]): size of the shift for the sliding window
        num_windows (int): number of windows

    Returns:
        Tensor: attention mask
    """
    # generate attention mask
    attn_mask = torch.zeros((pad_h, pad_w))
    h_slices = ((0, -shift_size[0]), (-shift_size[0], None))
    w_slices = ((0, -shift_size[1]), (-shift_size[1], None))
    count = 0
    for h in h_slices:
        for w in w_slices:
            attn_mask[h[0] : h[1], w[0] : w[1]] = count
            count += 1
    attn_mask = attn_mask.view(
        pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1]
    )
    attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
        num_windows, window_size[0] * window_size[1]
    )
    attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


def define_3d_relative_position_index(window_size: tuple[int]) -> Tensor:
    """Build the index for the relative positional bias of sliding attention
    windows in the context of 3D data.

    Args:
        window_size (tuple[int]): size of the sliding window

    Returns:
        Tensor: index
    """
    assert (
        len(window_size) == 3
    ), f"Data must be 3D, but window has {len(window_size)} dimension(s)"
    # generate 3D coordinates
    coords_z = torch.arange(window_size[0])
    coords_h = torch.arange(window_size[1])
    coords_w = torch.arange(window_size[2])
    coords = torch.stack(torch.meshgrid(coords_z, coords_h, coords_w, indexing="ij"))
    # flatten coordinates 3, Wz*Wh*Ww
    coords_flatten = torch.flatten(coords, 1)
    # build relative positions for each voxel ; 3, Wz*Wh*Ww, Wz*Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # rearrange relative positions to Wz*Wh*Ww, W*Wh*Ww, 3
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    # shift to start from 0
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 2] += window_size[2] - 1
    # shift z and h relative position because the positional bias is vector
    relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
    relative_coords[:, :, 1] *= 2 * window_size[2] - 1
    # and sum to get the index of that vector
    relative_position_index = relative_coords.sum(-1).flatten()

    return relative_position_index


def define_3d_earth_position_index(window_size: tuple[int]) -> Tensor:
    """Build the index for the Earth specific positional bias of sliding
    attention windows from PanguWeather.
    See http://arxiv.org/abs/2211.02556

    Args:
        window_size (tuple[int]): size of the sliding window

    Returns:
        Tensor: index
    """
    assert (
        len(window_size) == 3
    ), f"Data must be 3D, but window has {len(window_size)} dimension(s)"

    # Index in the pressure level of query matrix
    coords_zi = torch.arange(window_size[0])
    # Index in the pressure level of key matrix
    coords_zj = -torch.arange(window_size[0]) * window_size[0]

    # Index in the latitude of query matrix
    coords_hi = torch.arange(window_size[1])
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(window_size[1]) * window_size[1]

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(window_size[2])

    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
    coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += window_size[2] - 1
    coords[:, :, 1] *= 2 * window_size[2] - 1
    coords[:, :, 0] *= (2 * window_size[2] - 1) * window_size[1] * window_size[1]

    # Sum up the indexes in three dimensions
    position_index = coords.sum(dim=-1)

    # Flatten the position index to facilitate further indexing
    position_index = torch.flatten(position_index)

    return position_index


def slice_deserializer(value: str) -> slice:
    """Retrieve a slice from a string (serialized slice).
    Mandatory for jsonargparse to support
    slice in yaml config file.

    Args:
        value (str): serialized slice, e.g., "slice('2019', '2021')"

    Returns:
        slice
    """
    assert "slice" in value, "Not a slice, can't deserialize."
    arg_list = value[value.find("(") + 1 : value.find(")")].split(", ")
    assert len(arg_list) <= 3, "Slice takes 3 arguments at maximum."
    assert len(arg_list) > 0, "You must at least define the end of the slice."
    arg_list = [None if arg == "None" else arg.replace("'", "") for arg in arg_list]
    return slice(*arg_list)

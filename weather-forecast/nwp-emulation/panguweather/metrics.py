# Copyright The Lightning team.
#
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
from typing import Any

import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric


class LatitudeWeightedMeanSquaredError(Metric):
    r"""Compute `mean squared error`_ (MSE).

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mean_squared_error`` (:class:`~torch.Tensor`): A tensor with the mean squared error

    Args:
        squared: If True returns MSE value, if False returns RMSE value.
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        Single output mse computation:

        >>> from torch import tensor
        >>> from torchmetrics.regression import MeanSquaredError
        >>> target = tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = tensor([3.0, 5.0, 2.5, 7.0])
        >>> mean_squared_error = MeanSquaredError()
        >>> mean_squared_error(preds, target)
        tensor(0.8750)

    Example::
        Multioutput mse computation:

        >>> from torch import tensor
        >>> from torchmetrics.regression import MeanSquaredError
        >>> target = tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> preds = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        >>> mean_squared_error = MeanSquaredError(num_outputs=3)
        >>> mean_squared_error(preds, target)
        tensor([1., 4., 9.])

    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    sum_squared_error: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(squared, bool):
            raise ValueError(
                f"Expected argument `squared` to be a boolean but got {squared}"
            )
        self.squared = squared

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, latitudes: Tensor) -> None:
        """Update state with predictions and targets."""
        latitudes = latitudes[0] if latitudes.ndim > 1 else latitudes
        lat_weights = latitudes.shape[0] * (
            torch.cos(torch.pi * latitudes / 180)
            / torch.cos(torch.pi * latitudes / 180).sum()
        )
        # preds and targets are of shape B, lat, long
        squared_error = (preds - target) ** 2
        weighted_squared_error = squared_error * lat_weights.unsqueeze(0).unsqueeze(-1)
        swse = weighted_squared_error.mean(dim=(1, 2))
        self.sum_squared_error += swse.sum()
        self.total += len(preds)

    def compute(self) -> Tensor:
        """Compute mean squared error over state."""
        return (
            self.sum_squared_error / self.total
            if self.squared
            else torch.sqrt(self.sum_squared_error / self.total)
        )

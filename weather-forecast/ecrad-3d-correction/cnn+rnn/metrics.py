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

import numpy as np


def bias(y_true, y_pred, dim=0):
    """
    Compute the Bias between the ground truth and the predictions.
    """
    return np.mean(y_true - y_pred, axis=dim, dtype=np.float64)


def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error between the ground truth and the predictions.
    """
    return np.mean(np.abs(y_true - y_pred), axis=0, dtype=np.float64)


def rmse(y_true, y_pred, dim=0):
    """
    Compute the Root Mean Squared Error between the ground truth and the predictions.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=dim, dtype=np.float64))


def mape(y_true, y_pred, dim=0):
    """
    Compute the Mean Absolute Percentage Error between the ground truth and the predictions.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100, axis=dim, dtype=np.float64)


def percentiles(y_true, y_pred, percentiles=(5, 25, 50, 75, 95), dim=0):
    """
    Compute the percentiles of the differences between the ground truth and the predictions.
    """
    return np.percentile(y_true - y_pred, percentiles, axis=dim)

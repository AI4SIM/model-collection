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

from numpy.random import randint
from typing import Union


class RandomCropper3D():
    """Randomly crop a sub-block out of a 3D tensor.

    Args:
        out_shape (tuple or int): desired output shape.
    """

    def __init__(self, out_shape: Union[int, tuple]):
        assert isinstance(out_shape, (int, tuple))
        if isinstance(out_shape, int):
            self.out_shape = (out_shape, out_shape, out_shape)
        else:
            assert len(out_shape) == 3
            self.out_shape = out_shape

    def __call__(self, x, y):
        h, w, d = x.shape[0], x.shape[1], x.shape[2]
        block_h, block_w, block_d = self.out_shape
        tx = randint(0, h - block_h)
        ty = randint(0, w - block_w)
        tz = randint(0, d - block_d)
        x_cropped = x[tx:tx+block_h, ty:ty+block_w, tz:tz+block_d]
        y_cropped = y[tx:tx+block_h, ty:ty+block_w, tz:tz+block_d]
        return x_cropped, y_cropped

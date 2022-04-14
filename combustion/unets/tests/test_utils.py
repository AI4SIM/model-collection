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

from unittest import TestCase, main
from utils import RandomCropper3D
from numpy import copy
from numpy.random import rand


class TestData(TestCase):

    def test_random_cropper(self):
        n, n_ = 64, 32
        x = rand(n, n, n)
        y = copy(x)
        random_cropper = RandomCropper3D(n_)
        x_, y_ = random_cropper(x, y)
        self.assertEqual(x_.shape, (n_, n_, n_))
        self.assertEqual(y_.shape, (n_, n_, n_))
        self.assertEqual(x_[0, 0, 0], y_[0, 0, 0])


if __name__ == '__main__':
    main()

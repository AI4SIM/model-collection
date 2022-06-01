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
from numpy.random import rand
from torch import from_numpy
from samplers import Downsampler, Upsampler


class TestSamplers(TestCase):
    """Testing down- and upsamplers."""

    def test_downsampler(self):
        sampler = Downsampler(dim=1, inp_ch=4, out_ch=8).double()
        inp = from_numpy(rand(1, 4, 16))
        shp = sampler(inp).shape
        self.assertEqual(shp, (1, 8, 8))

    def test_upsampler(self):
        sampler = Upsampler(dim=1, inp_ch=8, out_ch=4).double()
        inp = from_numpy(rand(1, 8, 16))
        res = from_numpy(rand(1, 4, 32))
        shp = sampler(inp, res).shape
        self.assertEqual(shp, (1, 4, 32))  # last DoubleConv enforces the out_ch.


if __name__ == '__main__':
    main()

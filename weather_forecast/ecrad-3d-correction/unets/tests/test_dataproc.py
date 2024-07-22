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
import os.path as osp
from dataproc import ThreeDCorrectionDataproc
from tempfile import mkdtemp
from shutil import rmtree


class TestDataproc(TestCase):

    def setUp(self) -> None:
        """Create a temporary environment."""
        self.data_path = mkdtemp()
        self.dataproc = ThreeDCorrectionDataproc(
            self.data_path,
            timestep=3500,
            patchstep=16,
            num_workers=1)

    def tearDown(self) -> None:
        rmtree(self.data_path)

    def test_process(self) -> None:
        self.dataproc.process()
        self.assertTrue(osp.exists(osp.join(self.data_path, "processed", "stats.pt")))
        self.assertTrue(osp.exists(osp.join(self.data_path, "processed", "x")))
        self.assertTrue(osp.exists(osp.join(self.data_path, "processed", "y")))


if __name__ == '__main__':
    main()

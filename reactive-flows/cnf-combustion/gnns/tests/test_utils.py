"""This module proposes a test suite for the utils module."""

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

import unittest

from utils import create_graph_topo


class TestUtils(unittest.TestCase):
    """Utils function test suite."""

    def test_create_graph_topo(self):
        """Test the "create_graph_topo" function."""
        topo = create_graph_topo((10, 10, 10))
        self.assertEqual(topo.num_nodes, 1000)
        self.assertEqual(topo.pos.shape, (1000, 3))


if __name__ == "__main__":
    unittest.main()

"""This module proposes utils functions for the gnn use-case"""

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

from typing import Tuple

import networkx as nx
import torch_geometric as pyg


def create_graph_topo(grid_shape: Tuple[int, int, int]) -> pyg.data.Data:
    """Create a graph topology.

    Args:
        grid_shape (Tuple[int, int, int]): the shape of the grid for the
            z, y and x sorted dimensions.

    Return:
        (pyg.data.Data): the graph topology.
    """
    g0 = nx.grid_graph(dim=grid_shape)
    graph_topology = pyg.utils.convert.from_networkx(g0)
    graph_topology.grid_shape = grid_shape
    return graph_topology

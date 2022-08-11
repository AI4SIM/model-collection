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

from typing import Dict, List

import torch 
from torch.nn import Module, LazyConv1d, MultiheadAttention
from layers import PreProcessing, HRLayer


class Cnn1dAttention(Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.pre_processing = PreProcessing(...)
        self.block_dilation = CNBlockDilation(...)
        self.mha = MultiheadAttention(...)
        self.block_regular = CNBlockRegular(...)
        self.conv1d = LazyConv1d(...)
        self.hr_layer = HRLayer(...)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass
    
class CNBlockDilation(Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
class CNBlockRegular(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
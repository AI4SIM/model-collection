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

"""
This module simply load all the nox targets defined in the reference noxfile
and make them available for the use-case.
This file can be enriched by use case specific targets.
"""

import os
import sys
import inspect

# Insert the tools/nox folder to the python path to fetch the nox_ref_file.py content
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
common_dir = os.path.dirname(os.path.dirname(current_dir))
build_ref_dir = os.path.join(common_dir, "tools", "nox")
sys.path.insert(0, build_ref_dir)

# Fetch the nox_ref_file.py content
from nox_ref_file import *

# Insert below the use case specific targets

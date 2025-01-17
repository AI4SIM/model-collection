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

import os
import sys
from os import makedirs
from os.path import exists, join

import numpy as np
from h5py import File
from yaml import dump

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
data_path = "./data"


def create_data():
    """Create data folder with fake raw data"""
    filenames = ["test_1.h5", "test_2.h5", "test_3.h5"]

    if not exists(data_path):
        makedirs(join(data_path, "raw"))
        for file_h5 in filenames:
            with File(join(data_path, "raw", file_h5), "w") as f:
                f["filt_8"] = np.random.normal(0, 1, (320, 160, 160))
                f["filt_grad_8"] = np.random.normal(0, 1, (320, 160, 160))
                f["grad_filt_8"] = np.random.normal(0, 1, (320, 160, 160))

        temp_file_path = join(data_path, "filenames.yaml")
        with open(temp_file_path, "w") as tmpfile:
            dump(filenames, tmpfile)
    else:
        raise Exception(f"Remove manually {data_path}")


if __name__ == "__main__":

    create_data()

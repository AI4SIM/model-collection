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

import config  # noqa:


def create_data():
    """Create data folder with fake raw data"""
    filenames = [
        "test_1.h5",
        "test_2.h5",
        "test_3.h5",
        "test_4.h5",
        "test_5.h5",
        "test_6.h5",
        "test_7.h5",
    ]

    if not exists(config.data_path):
        makedirs(join(config.data_path, "raw"))
        for file_h5 in filenames:
            with File(join(config.data_path, "raw", file_h5), "w") as f:
                f["/c_filt"] = np.random.normal(0, 1, (20, 20, 20))
                f["/c_grad_filt"] = np.random.normal(0, 1, (20, 20, 20))

        temp_file_path = join(config.data_path, "filenames.yaml")
        with open(temp_file_path, "w") as tmpfile:
            dump(filenames, tmpfile)
    else:
        raise Exception(f"Remove manually {config.data_path}")


if __name__ == "__main__":

    create_data()

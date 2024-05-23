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

from os.path import join, exists
from os import makedirs, listdir, environ
from h5py import File
from yaml import dump, safe_load
from tempfile import mkdtemp
from shutil import rmtree
from numpy import zeros


def read_config(file: str) -> str:
    """Read the data fake config file"""
    with open(file, 'r') as file:
        data = safe_load(file)
         
    return data['data']['test_path']

def create_data(tempdir):
    """Create data folder with fake raw data"""
    filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
    
    if (not exists(tempdir)):
        makedirs(join(tempdir, "data","raw"))
    folder = join(tempdir, "data", "raw")

    for file_h5 in filenames:
        with File(join(tempdir, "data", "raw", file_h5), 'w') as f:
            f['filt_8'] = zeros((10, 10, 10))
            f['filt_grad_8'] = zeros((10, 10, 10))
            f['grad_filt_8'] = zeros((10, 10, 10))

    temp_file_path = join(tempdir, 'data', 'filenames.yaml')
    with open(temp_file_path, 'w') as tmpfile:
        dump(filenames, tmpfile)
    
    file = join(tempdir, 'data', 'filenames.yaml')

if __name__ == '__main__':
    file = 'tests/configs/data.yaml'
    tempdir = read_config(file)
    #tempdir = mkdtemp(prefix="test") 
    create_data(tempdir)
    #rmtree(tempdir)
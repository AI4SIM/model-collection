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

from os.path import join, exists, dirname,  realpath
from os import makedirs, listdir
from h5py import File
from yaml import dump, safe_load
from tempfile import mkdtemp
from shutil import rmtree, move
from numpy import zeros


def create_data(tempdir):
    """Create data folder with fake raw data"""
    filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
    
    if (not exists(join(tempdir,'data'))):
        makedirs(join(tempdir, "data","raw"))
    folder = join(tempdir, "data", "raw")

    for file_h5 in filenames:
        with File(join(tempdir, "data", "raw", file_h5), 'w') as f:
            f['filt_8'] = zeros((320, 160, 160))
            f['filt_grad_8'] = zeros((320, 160, 160))
            f['grad_filt_8'] = zeros((320, 160, 160))

    temp_file_path = join(tempdir, 'data', 'filenames.yaml')
    with open(temp_file_path, 'w') as tmpfile:
        dump(filenames, tmpfile)
    
    file = join(tempdir, 'data', 'filenames.yaml')

if __name__ == '__main__':
    
    tempdir = dirname(realpath(__file__))
    create_data(tempdir)
    dst = "/".join(tempdir.split("/")[:-1])

    # Before moving check if dst folder exists
    if (exists(join(dst,'data'))):
         rmtree(join(dst,'data'))
    move(join(tempdir,'data'), dst)    
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
from os.path import join, exists
from os import mkdir, listdir
from h5py import File
#from data import CnfCombustionDataset, CnfCombustionDataModule
from yaml import dump
from tempfile import mkdtemp
from shutil import rmtree
from numpy import zeros
#from torch.utils.data import DataLoader
#from warnings import catch_warnings, simplefilter


class TestData(TestCase):

    def setUp(self) -> None:
        """Generate fake data from the real data footprint"""
        self.filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
        self.data_module_params = {
            'batch_size': 1,
            'num_workers': 0,
            'y_normalizer': 342.553,
            'splitting_lengths': [1, 1, 1],
            'subblock_shape': (32, 16, 16)}

        # Creates a temporary environment.
        self.dir = mkdtemp()
        self.create_env(self.dir)

    def tearDown(self) -> None:
      """Clean the folder tree when the test is over"""
        rmtree(self.dir)

    def create_env(self, tempdir):
      """Create data folder with the raw data"""
        mkdir(join(tempdir, "data"))
        
        # Check data folder creation
        folder = join(tempdir, "data")
        self.assertTrue(folder, f"{fodler} has been correctly created")
        
        mkdir(join(tempdir, "data", "raw"))
        
        # Check data raw folder creation
        folder = join(tempdir, "data", "raw")
        self.assertTrue(folder, f"{fodler} has been correctly created")

        for file_h5 in self.filenames:
            with File(join(tempdir, "data", "raw", file_h5), 'w') as f:
                f['filt_8'] = zeros((10, 10, 10))
                f['filt_grad_8'] = zeros((10, 10, 10))
                f['grad_filt_8'] = zeros((10, 10, 10))

        temp_file_path = join(tempdir, 'data', 'filenames.yaml')
        with open(temp_file_path, 'w') as tmpfile:
            dump(self.filenames, tmpfile)
            
        # Check filenames.yaml creation     
        file = join(tempdir, 'data', 'filenames.yaml')
        self.assertTrue(file, f"{file} has been correctly created")    

        

if __name__ == '__main__':
    main()

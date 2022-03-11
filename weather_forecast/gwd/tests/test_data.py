"""This module provides a unit tests suite for the data.py module."""

import os
import shutil
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import h5py
import torch

import yaml

from data import NOGWDDataset, NOGWDDataModule

TEST_DATA_PATH = os.path.join("test_data", "data")
REF_FILENAMES_FILE = os.path.join("test_data", "filenames-split.yaml")


def get_filenames(filenames_file):
    """ Extract the list of file that are required for test, from the input file.

    Args:
        filenames_file (str): the path of filenames file

    Returns:
        filenames (list): the filenames extracted from the input file.
    """
    with open(filenames_file, 'r') as file:
        yaml_content = yaml.safe_load(file)
        filenames = []
        for files in yaml_content.values():
            filenames.extend(files)
    return filenames


def populate_test_data(filenames) -> None:
    """ Create fake random data file.

    Args:
        filenames (list): the list of filenames to be created.
    """
    data_path = os.path.join(TEST_DATA_PATH, "raw")
    Path(data_path).mkdir(parents=True, exist_ok=True)

    for file_h5 in filenames:
        with h5py.File(os.path.join(data_path, file_h5), 'w') as file:
            file['/x'] = np.random.rand(191, 36, 10)
            file['/y'] = np.random.rand(126, 36, 10)


class TestNOGWDDatasetTrain(unittest.TestCase):
    """Test the NOGWDDataset class in train mode."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set-up the test environment to get the data files required to test the NOGWDDataset
        class."""
        # get the list of filename that should be created
        filenames = get_filenames(REF_FILENAMES_FILE)
        # create the required files with fake data
        populate_test_data(filenames)
        # put a copy of the 'filenames-split.yaml' file in the expected place
        shutil.copy(REF_FILENAMES_FILE, TEST_DATA_PATH)

    def setUp(self) -> None:
        """Instantiate the NOGWDDataset class in train mode."""
        self.data_test = NOGWDDataset(root=TEST_DATA_PATH, mode='train')

    def test__compute_stats(self):
        """Test the instantiated NOGWDDataset class in train mode, has created the stat file."""
        self.assertTrue(os.path.isfile(os.path.join(TEST_DATA_PATH, 'stats.pt')))

    def test_raw_filenames_property(self):
        """Test the 'raw_filenames' property is properly set with the train dataset files."""
        expected_file_list = ['2015-01-01.h5', '2016-01-01.h5', '2017-01-01.h5']
        self.assertListEqual(self.data_test.raw_filenames, expected_file_list)

    def test_download(self):
        """Test 'download' method is not yet implemented."""
        self.assertRaisesRegex(NotImplementedError,
                               "The 'download' method is not yet available",
                               self.data_test.download)

    def test_load(self):
        """Test the 'load' method returns tensors with good shapes."""
        x, y = self.data_test.load()
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        # 3 : nb of train .h5 files
        # 36 and 10 : dimension of input data
        # 191 and 126 : nb of feature respectively for x and y
        self.assertEqual(x.size(), (3*36*10, 191))
        self.assertEqual(y.size(), (3*36*10, 126))

    def test_get(self):
        """Test the '__getitem__' method returns the proper (x, y) tuple of tensors."""
        data_get = self.data_test[1]
        np.testing.assert_array_equal(data_get[0].numpy(), self.data_test.x[1, :].numpy())
        np.testing.assert_array_equal(data_get[1].numpy(), self.data_test.y[1, :].numpy())

    def test_len(self):
        """Test the '__len__' method returns the proper value.
        FIXME: note for now the 'shard_len' is hardcoded, but should be dynamically set."""
        data_len = len(self.data_test)
        # 3 : nb of train .h5 files
        self.assertEqual(data_len, 3*self.data_test.shard_len)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)


class TestNOGWDDatasetTest(unittest.TestCase):
    """Test the NOGWDDataset class in test mode."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set-up the test environment to get the data files required to test the NOGWDDataset
        class."""
        # get the list of filename that should be created
        filenames = get_filenames(REF_FILENAMES_FILE)
        # create the required files with fake data
        populate_test_data(filenames)
        # put a copy of the 'filenames-split.yaml' file in the expected place
        shutil.copy(REF_FILENAMES_FILE, TEST_DATA_PATH)

    def setUp(self) -> None:
        """Instantiate the NOGWDDataset class in test mode."""
        self.data_test = NOGWDDataset(root=TEST_DATA_PATH, mode='test')

    def test__compute_stats(self):
        """Test the instantiated NOGWDDataset class in train mode, has NOT created the stat file."""
        self.assertFalse(os.path.isfile(os.path.join(TEST_DATA_PATH, 'stats.pt')))

    def test_raw_filenames_property(self):
        """Test the 'raw_filenames' property is properly set with the train dataset files."""
        expected_file_list = ['2017-02-19.h5']
        self.assertListEqual(self.data_test.raw_filenames, expected_file_list)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)


class TestNOGWDDatasetVal(unittest.TestCase):
    """Test the NOGWDDataset class in val mode."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set-up the test environment to get the data files required to test the NOGWDDataset
        class."""
        # get the list of filename that should be created
        filenames = get_filenames(REF_FILENAMES_FILE)
        # create the required files with fake data
        populate_test_data(filenames)
        # put a copy of the 'filenames-split.yaml' file in the expected place
        shutil.copy(REF_FILENAMES_FILE, TEST_DATA_PATH)

    def setUp(self) -> None:
        """Instantiate the NOGWDDataset class in val mode."""
        self.data_test = NOGWDDataset(root=TEST_DATA_PATH, mode='val')

    def test__compute_stats(self):
        """Test the instantiated NOGWDDataset class in train mode, has NOT created the stat file."""
        self.assertFalse(os.path.isfile(os.path.join(TEST_DATA_PATH, 'stats.pt')))

    def test_raw_filenames_property(self):
        """Test the 'raw_filenames' property is properly set with the train dataset files."""
        expected_file_list = ['2016-02-25.h5']
        self.assertListEqual(self.data_test.raw_filenames, expected_file_list)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)


class TestNOGWDDataModule(unittest.TestCase):
    """Test the NOGWDDataModule class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set-up the test environment to get the data files required to test the NOGWDDataset
        class."""
        # get the list of filename that should be created
        filenames = get_filenames(REF_FILENAMES_FILE)
        # create the required files with fake data
        populate_test_data(filenames)
        # put a copy of the 'filenames-split.yaml' file in the expected place
        shutil.copy(REF_FILENAMES_FILE, TEST_DATA_PATH)

    def setUp(self) -> None:
        """Instantiate the NOGWDDataset class in val mode."""
        self.data_module = NOGWDDataModule(batch_size=1, num_workers=0)

    @patch("config.data_path", TEST_DATA_PATH)
    def test_setup_fit(self):
        """Test the 'setup' method set properly the train and val attributes in 'fit' mode."""
        self.data_module.setup("fit")
        self.assertIsInstance(self.data_module.train, NOGWDDataset)
        self.assertIsInstance(self.data_module.val, NOGWDDataset)
        self.assertIsNone(self.data_module.test)

    @patch("config.data_path", TEST_DATA_PATH)
    def test_setup_test(self):
        """Test the 'setup' method set properly the test attribute in 'test' mode."""
        self.data_module.setup("test")
        self.assertIsNone(self.data_module.train)
        self.assertIsNone(self.data_module.val)
        self.assertIsInstance(self.data_module.test, NOGWDDataset)

    @patch("config.data_path", TEST_DATA_PATH)
    def test_train_dataloader(self):
        """Test the 'train_dataloader' method properly instantiate a DataLoader."""
        self.data_module.setup("fit")
        dataloader = self.data_module.train_dataloader()
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

    @patch("config.data_path", TEST_DATA_PATH)
    def test_val_dataloader(self):
        """Test the 'val_dataloader' method properly instantiate a DataLoader."""
        self.data_module.setup("fit")
        dataloader = self.data_module.val_dataloader()
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

    @patch("config.data_path", TEST_DATA_PATH)
    def test_test_dataloader(self):
        """Test the 'test_dataloader' method properly instantiate a DataLoader."""
        self.data_module.setup("fit")
        dataloader = self.data_module.test_dataloader()
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

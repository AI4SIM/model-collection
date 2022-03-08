"""
Test models file
"""

import unittest
import os
import tempfile
import h5py
import numpy as np
import yaml
import warnings
import networkx as nx
import torch
import torch_geometric as pyg
import models
import torch_optimizer as optim

class TestData(unittest.TestCase):
    """
    model test file
    """
    
    def setUp(self) -> None:
        """
        define default parameters
        """
        
        self.filenames = ['DNS1_00116000.h5', 'DNS1_00117000.h5', 'DNS1_00118000.h5']
        
        self.initParam ={'in_channels': 1, 'hidden_channels': 32, 'out_channels': 1, 
                        'num_layers': 4, 'dropout': .5, 'jk': "last", 'lr': .0001}
        
        
    def create_env(self, tempdir):
        
        os.mkdir(os.path.join(tempdir,"data"))
        os.mkdir(os.path.join(tempdir,"data", "raw"))

        for file_h5 in self.filenames:
            with h5py.File(os.path.join(tempdir,"data", "raw",file_h5), 'w') as f:
                f['filt_8']      = np.zeros((10, 10, 10))
                f['filt_grad_8'] = np.zeros((10, 10, 10))
                f['grad_filt_8'] = np.zeros((10, 10, 10))

        temp_file_path = os.path.join(tempdir, 'data', 'filenames.yaml')
        with open(temp_file_path, 'w') as tmpfile:
            documents = yaml.dump(self.filenames, tmpfile)
            
    def create_graph(self, file):
        with h5py.File(file, 'r') as file:
            c = file["/filt_8"][:]
            sigma = file["/filt_grad_8"][:]
        
        x_size, y_size, z_size = c.shape
        grid_shape = (z_size, y_size, x_size)

        g0 = nx.grid_graph(dim=grid_shape)
        graph = pyg.utils.convert.from_networkx(g0)
        undirected_index = graph.edge_index
        coordinates = list(g0.nodes())
        coordinates.reverse()

        data = pyg.data.Data(
            x=torch.tensor(c.reshape(-1,1), dtype=torch.float), 
            edge_index=torch.tensor(undirected_index, dtype=torch.long),
            pos=torch.tensor(np.stack(coordinates)),
            y=torch.tensor(sigma.reshape(-1,1), dtype=torch.float))

        return data
    
    def create_data(self, repeat):
        data = []
        for r in range(repeat):
            x = torch.tensor([1, 2, 3], dtype=torch.float)
            e = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
            s = f'{r+1}'
            data.append(pyg.data.Data(x, e, s=s))
        
        return data
        
        
    def test_forward(self):
    
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir,"data", "raw", self.filenames[0])
            data_test = self.create_graph(file_path)
            
            test_gcn = models.LitGCN(**self.initParam)
            test_forward = test_gcn.forward(data_test.x, data_test.edge_index)
            
            self.assertTrue(isinstance(test_forward, torch.Tensor))
        
    def test_common_step(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir,"data", "raw",self.filenames[0])
            data_test = self.create_graph(file_path)
            
            test_gcn = models.LitGCN(**self.initParam)
            batch =  pyg.data.Batch.from_data_list([data_test, data_test])
            
            loss = test_gcn._common_step(batch = batch, batch_idx = 1, stage= "train")
            
            self.assertTrue(len(loss)==3)
            
    def test_training_step(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir,"data", "raw",self.filenames[0])
            data_test = self.create_graph(file_path)
            
            test_gcn = models.LitGCN(**self.initParam)
            batch =  pyg.data.Batch.from_data_list([data_test, data_test])
            
            loss = test_gcn.training_step(batch = batch, batch_idx = 1)
            self.assertTrue(isinstance(loss, torch.Tensor))
            
#     def test_validation_step(self):
        
#         with tempfile.TemporaryDirectory() as tempdir:
#             self.create_env(tempdir)
#             file_path = os.path.join(tempdir,"data", "raw",self.filenames[0])
#             data_test = self.create_graph(file_path)
            
#             test_gcn = models.LitGCN(**self.initParam)
#             batch =  pyg.data.Batch.from_data_list([data_test, data_test])
            
#             y_hat = test_gcn.validation_step(batch = batch, batch_idx = 1)
            
#             print(y_hat)
#             self.assertTrue(False)
#             self.assertTrue(isinstance(y_hat, torch.Tensor))
            
    def test_test_step(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir,"data", "raw",self.filenames[0])
            data_test = self.create_graph(file_path)
            
            test_gcn = models.LitGCN(**self.initParam)
            batch =  pyg.data.Batch.from_data_list([data_test, data_test])
            
            t = test_gcn.test_step(batch = batch, batch_idx = 1)
            
            self.assertTrue(len(t)==2)
            self.assertEqual(t[0].size(),t[0].size())
            
            
    def test_configure_optimizers(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            self.create_env(tempdir)
            file_path = os.path.join(tempdir,"data", "raw",self.filenames[0])
            data_test = self.create_graph(file_path)
            
            test_gcn = models.LitGCN(**self.initParam)
            
            op = test_gcn.configure_optimizers()
            
            self.assertTrue(isinstance(op, optim.Optimizer))
    
    
if __name__ == '__main__':
    unittest.main()
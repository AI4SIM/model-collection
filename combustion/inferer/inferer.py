import numpy as np
import torch
import torch_geometric as pyg
import time
import h5py
import networkx as nx


class inference_pth_gnn:
    
    def __init__(self, file_path):
        print("GNN model (torch + torch_geometric)")
        # self.y_normalizer = 3295.4
        
        
        with h5py.File(file_path, 'r') as file:
            self.c = file["/c_filt"][:]

            # self.sigma = file["/filt_grad_8"][:]
            # if self.y_normalizer is not None :
            #     self.sigma /= self.y_normalizer
        
        
    # Load model and first prediction
    def init_nn(self, model_file):
        self.model = torch.load(model_file, map_location="cuda:3")
        #self.model = torch.load(model_file, map_location="cpu")
        # self.create_graph()
        
    def create_graph(self):
        
        x_size, y_size, z_size = self.c.shape
        grid_shape = (z_size, y_size, x_size)

        g0 = nx.grid_graph(dim=grid_shape)
        graph = pyg.utils.convert.from_networkx(g0)
        undirected_index = graph.edge_index
        coordinates = list(g0.nodes())
        coordinates.reverse()

        data = pyg.data.Data(
            x=torch.tensor(self.c.reshape(-1,1), dtype=torch.float), 
            edge_index=torch.tensor(undirected_index, dtype=torch.long)
        )
        
        return data
    
    def prediction(self):
        
        data = self.create_graph(self.c)
        
        gin = pyg.data.Batch().from_data_list(data_list=[data]) 
        
        # with torch.no_grad():
        self.model.eval()
        gin.to(self.cuda_device)
        output = self.model(gin.x, gin.edge_index)
        
        
if __name__ == "__main__" :
    inferer = inference_pth_gnn(file_path='R3_data/smaller_new_filt_15_F_4_cropped_progvar_R3.h5')
    inferer.init_nn('gnn_model/logs/version_0/checkpoints/epoch=999-step=70999.ckpt')
    inferer.create_graph()
    
    # inferer.prediction()
    

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
import pickle
import logging
import h5py
import networkx as nx
import torch
import torch_geometric as pyg
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI


class InferencePthGnn:
    """The inference class used to infer the data provided through a file with the model provided
    through a checkpoint save.
    """
    def __init__(self,
                 model_path: str,
                 data_path: str,
                 model_class: LightningModule = None,
                 wkd: str = os.getcwd()):
        """Init the InferencePthGnn class with additional args required (y_normalizer, model_class).
        The model is automatically loaded from the checkpoint during the initialization.

        Args:
            model_path (str): the path of the model checkpoint.
            data_path (str): the path of the input data to be infered.
            model_class (LightningModule): the class that as generated the saved model.
            wkd (str): the path of the inference working directory were the data could be saved.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model_class = model_class

        self.wkd = wkd

        # init the additional attributes to None
        self.model = None
        self.data = None

        # Load the NN from self.model_path file
        logging.info(f"Start loading NN model from {self.model_path} ...")
        self.load_nn()
        logging.info("NN model loaded")

    def load_nn(self):
        """Load the NN model from a checkpoint. That requires to restores the model states in the
        class that generated the model (provided as input of the class, "model_class")."""
        self.model = self.model_class.load_from_checkpoint(self.model_path)
        self.model.eval()

    def _load_data(self):
        """Load the input data that will be injected in the model to process the inference."""
        with h5py.File(self.data_path, 'r') as file:
            feat = file["/c_filt"][:]
            logging.info("Data input R3 loaded")
            return feat

    def _load_y_dns(self):
        """Load the ground truth data provided in the input file, to be used in a post inference
        validation step. These data corresponds to the DNS ones."""
        with h5py.File(self.data_path, 'r') as file:
            y_gt = file["/c_grad_filt"][:]
            logging.info("Data ground truth R3 loaded")
            return y_gt

    def _load_y_les(self):
        """Load additional data provided in the input file, to be used in a post inference
        validation step. These data corresponds to the LES ones."""
        with h5py.File(self.data_path, 'r') as file:
            y_les = file["/c_filt_grad"][:]
            logging.info("Data ground truth LES loaded")
            return y_les

    @property
    def data_processed_path(self):
        """Provide a path to store the preprocessed data, inorder to save time between successive
        inference runs on the same data."""
        path_name = os.path.splitext(os.path.basename(self.data_path))[0] + '.data'
        return os.path.join(self.wkd, path_name)

    def _create_graph(self, feat: torch.Tensor, save: bool=False):
        """Format the input data as a graph to be provided to the GNN. The processed data can be
        saved in a pickle file, if "save" is True.

        Args:
            feat (torch.Tensor): the input data to be formatted as a graph.
            save (bool): if the processed data are saved in a pickle file, or not. Default: False.
        """
        try:
            with open(self.data_processed_path, 'rb') as file:
                self.data = pickle.load(file)

        except FileNotFoundError:
            x_size, y_size, z_size = feat.shape
            grid_shape = (z_size, y_size, x_size)
            g_0 = nx.grid_graph(dim=grid_shape)
            graph = pyg.utils.convert.from_networkx(g_0)
            undirected_index = graph.edge_index

            self.data = pyg.data.Data(
                x=torch.tensor(feat.reshape(-1, 1), dtype=torch.float),
                edge_index=torch.tensor(undirected_index, dtype=torch.long)
            )

            if save:
                with open(self.data_processed_path, 'wb') as file:
                    pickle.dump(self.data, file)

    def preprocess(self, save: bool = False):
        """Preprocess the input data to provide the self.data.
        
        Args:
            save (bool): If the preprocessed data will be saved in a file or not. Default, False.
        """
        features = self._load_data()
        self._create_graph(features, save=save)

    def predict(self):
        """Run the inference of self.data on the restored model."""
        return self.model(self.data.x, self.data.edge_index)


if __name__ == "__main__":
    from data import CombustionDataset  # noqa:  registry should be imported to
    cli = LightningCLI(run=False)

    inferer = InferencePthGnn(
        model_path="exp_save/gin_1000epoch_0.96r2/offline-burrito/logs/version_0/checkpoints/epoch=999-step=35999.ckpt",  # noqa:
        data_path='/net/172.16.118.188/data/raise/R2_flame/combustiondatabase/R2-filtered/R3-data/smaller_new_filt_15_F_4_cropped_progvar_R3.h5',
        model_class=cli.model.__class__
    )

    logging.info("Start creating data from input data ...")
    inferer.preprocess()
    logging.info("Input data data created")

    logging.info("Start inference on input data ...")
    output = inferer.predict()
    logging.info(f"Inference result:\n {output}")

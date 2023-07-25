"""This module proposes Pytorch style LightningModule classes for the 3dcorrection use-case."""
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
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.utils import scatter
import torch_optimizer as optim
import torchmetrics.functional as tmf
from typing import List, Tuple, Optional


class ThreeDCorrectionModule(pl.LightningModule):
    """Contain the mutualizable model logic for the 3dcorrection use-case."""

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute the forward pass.
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Connectivity matrix.
            edge_attr (torch.Tensor): Edge features.
            u: (torch.Tensor): Global graph features.
            batch (torch.Tensor): Batch vector which assigns each node
                to a specific graph.

        Returns:
            (torch.Tensor): Resulting model forward pass for node features.
            (Optional[torch.Tensor]): Resulting model forward for edge features.
            (Optional[torch.Tensor]): Resulting model forward for global graph features.
        """
        return self.net(x, edge_index, edge_attr, u, batch)

    def _common_step(
        self, batch: torch.Tensor, batch_idx: int, stage: str
    ) -> List[torch.Tensor]:
        """Define the common operations performed on data.

        Args:
            batch (torch.Tensor): The output of the DataLoader.
            batch_idx (int): Integer displaying index of this batch.
            stage (str): current running stage (fit/val/test).

        Returns:
            (List[torch.Tensor]): predictions and metrics
        """
        (x, y, z, edge_index, edge_attr, u, batch_) = (
            batch.x,
            batch.y,
            batch.z,
            batch.edge_index,
            batch.edge_attr,
            batch.u,
            batch.batch,
        )

        y_node_hat, y_edge_hat, _ = self(x, edge_index, edge_attr, u, batch_)

        flux_loss = tmf.mean_squared_error(y_node_hat, y)
        hr_loss = tmf.mean_squared_error(y_edge_hat[::2], z)
        loss = flux_loss + hr_loss

        flux_mae = tmf.mean_absolute_error(y_node_hat, y)
        hr_mae = tmf.mean_absolute_error(y_edge_hat[::2], z)
        mae = flux_mae + hr_mae

        values = {
            f"{stage}_loss": loss,
            f"{stage}_flux_loss": flux_loss,
            f"{stage}_hr_loss": hr_loss,
            f"{stage}_mae": mae,
            f"{stage}_flux_mae": flux_mae,
            f"{stage}_hr_mae": hr_mae,
        }
        self.log_dict(
            values,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=batch.batch.max()+1
        )

        return y_node_hat, y_edge_hat, loss, mae

    def training_step(self, batch, batch_idx):
        """Compute one training step.
        Args:
            batch (torch.Tensor): Batch containing nodes features
                and connectivity matrix.
            batch_idx (int): Integer displaying index of this batch.
        Returns:
            (torch.Tensor): Loss.
        """
        _, _, loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute one validation step.

        Args:
            batch (torch.Tensor): Batch containing nodes features
                and connectivity matrix.
            batch_idx (int): Batch index.
        """
        _, _, _, _ = self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Compute one testing step.
        Args:
            batch (torch.Tensor): Batch containing x input
                and y output features.
            batch_idx (int): Batch index.
        """
        _, _, _, _ = self._common_step(batch, batch_idx, "test")


@MODEL_REGISTRY
class LitMeta(ThreeDCorrectionModule):
    """
    PyG implementation of the Graph Network described in 
    Battaglia et al. (https://arxiv.org/pdf/1806.01261.pdf)
    GlobalModel is implemented but not used as we are not focused on global
    graph features.
    """

    class EdgeModel(nn.Module):
        def __init__(self, edge_in_feat, edge_out_feat, edge_hidden):
            super().__init__()

            self.edge_in_feat = edge_in_feat
            self.edge_out_feat = edge_out_feat
            self.edge_hidden = edge_hidden

            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_in_feat, self.edge_hidden),
                nn.ELU(),
                nn.Linear(self.edge_hidden, self.edge_out_feat),
            )

        def forward(self, src, dst, edge_attr, u, batch):
            out = torch.cat([src, dst, edge_attr, u[batch]], dim=1)
            # print(f"[EdgeModel] - Output shape after concatenation : {out.shape}")
            return self.edge_mlp(out)

    class NodeModel(nn.Module):
        def __init__(self, node_in_feat, node_out_feat, node_hiddens):
            super().__init__()

            self.node_in_feat = node_in_feat
            self.node_out_feat = node_out_feat
            self.node_hiddens = node_hiddens

            assert len(self.node_hiddens) == 4

            self.node_mlp_1 = nn.Sequential(
                nn.Linear(self.node_in_feat, self.node_hiddens[0]),
                nn.ELU(),
                nn.Linear(self.node_hiddens[0], self.node_hiddens[1]),
            )
            self.node_mlp_2 = nn.Sequential(
                nn.Linear(self.node_hiddens[2], self.node_hiddens[3]),
                nn.ELU(),
                nn.Linear(self.node_hiddens[3], self.node_out_feat),
            )

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            # print(f"[NodeModel] - Output shape after concatenation : {out.shape}")
            out = self.node_mlp_1(out)
            out = scatter(out, col, dim=0, dim_size=x.size(0), reduce="mean")
            # print(f"[NodeModel] - Output shape after scatter : {out.shape}")
            out = torch.cat([x, out, u[batch]], dim=1)
            # print(f"[NodeModel] - Output shape after concatenation : {out.shape}")
            return self.node_mlp_2(out)

    class GlobalModel(nn.Module):
        def __init__(self, global_in_feat, global_out_feat, global_hidden):
            super().__init__()

            self.global_in_feat = global_in_feat
            self.global_out_feat = global_out_feat
            self.global_hidden = global_hidden

            self.global_mlp = nn.Sequential(
                nn.Linear(self.global_in_feat, self.global_hidden),
                nn.ELU(),
                nn.Linear(self.global_hidden, self.global_out_feat),
            )

        def forward(self, x, edge_index, edge_attr, u, batch):
            out = torch.cat(
                [
                    u,
                    scatter(x, batch, dim=0, reduce="mean"),
                ],
                dim=1,
            )
            print(
                f"[GlobalModel] - Output shape after scatter and concat : {out.shape}"
            )
            return self.global_mlp(out)

    def __init__(
        self,
        edge_in_feat,
        edge_out_feat,
        edge_hidden,
        node_in_feat,
        node_out_feat,
        node_hiddens,
        global_in_feat,
        global_out_feat,
        global_hidden,
        lr,
    ):
        """Init the LitMeta class."""
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.edge_model = self.EdgeModel(edge_in_feat, edge_out_feat, edge_hidden)

        self.node_model = self.NodeModel(node_in_feat, node_out_feat, node_hiddens)

        self.global_model = self.GlobalModel(
            global_in_feat, global_out_feat, global_hidden
        )

        # self.net = pyg.nn.models.MetaLayer(
        #     self.edge_model, self.node_model, self.global_model
        # )
        self.net = pyg.nn.models.MetaLayer(
            edge_model=self.edge_model, node_model=self.node_model, global_model=None
        )

    def configure_optimizers(self) -> optim.Optimizer:
        """Set the model optimizer.
        Returns:
            (torch_optimizer.Optimizer): Optimizer
        """
        return optim.AdamP(self.parameters(), lr=self.lr)

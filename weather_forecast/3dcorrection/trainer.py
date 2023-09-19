"""This module proposes Pytorch style a Trainer class for the 3dcorrection use-case."""
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

import json
import os
import itertools
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from typing import List, Union, Dict
import onnx
import mlflow
from mlflow.models.signature import infer_signature, ModelSignature

import config
import data  # noqa: F401 'data' imported but not used
import models  # noqa: F401 'data' imported but not used


class Trainer(pl.Trainer):
    """
    Modified PyTorch Lightning Trainer that automatically tests, logs, and
    writes artifacts by the end of training.
    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator, None],
        devices: Union[List[int], str, int, None],
        strategy: Union[str, Strategy, None],
        max_epochs: int,
        # TODO: delete.
        # For some reason, those two are mandatory in current version of Lightning.
        fast_dev_run: Union[int, bool] = False,
        callbacks: Union[List[Callback], Callback, None] = None,
        mlflow_setup: Dict = None
    ) -> None:
        """
        Args:
            accelerator (Union[str, pl.accelerators.Accelerator, None]): Type of accelerator
            to use for training.
            devices: (Union[List[int], str, int, None]): Devices explicit names
            to use for training.
            max_epochs (int): Maximum number of epochs if no early stopping logic is implemented.
        """
        if mlflow_setup is not None:
            self.with_mlflow = True
            # MLFlow set-up
            if mlflow_setup.get('git', None):
                if mlflow_setup['git'].get('git_python_refresh', None):
                    # Only required if you do not have git on the training platform.
                    os.environ['GIT_PYTHON_REFRESH'] = mlflow_setup['git']['git_python_refresh']
            # According to the way your tracking server has been deployed, you will be required to
            # provide artifact access credentials, through additional environment variables
            # (e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MLFLOW_S3_ENDPOINT_URL for an S3
            # artifact repository).
            # To avoid using credentials please consider deploying your MLFlow tracking server has
            # documented here in scenario 5 or 6 : https://mlflow.org/docs/latest/tracking.html.
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_setup['tracking']['uri']
            if mlflow_setup['tracking'].get('experiment_name', None):
                os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_setup['tracking']['experiment_name']
            self.model_name = mlflow_setup['tracking']['model_name']
        else:
            self.with_mlflow = False

        if accelerator == "cpu":
            devices = None

        logger = pl.loggers.TensorBoardLogger(config.logs_path, name=None)
        super().__init__(
            default_root_dir=config.logs_path,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=max_epochs,
            # TODO: for some reason, a forward pass happens in the model before datamodule creation.
            num_sanity_val_steps=0,
        )

    def test(self, **kwargs) -> None:
        """
        Use superclass test results, but additionally, saves raw results as a JSON file,
        and stores the model weights for future use in inference mode.

        Returns:
            None.
        """
        results = super().test(**kwargs)[0]

        with open(os.path.join(config.artifacts_path, "results.json"), "w") as f:
            json.dump(results, f)

        torch.save(self.model.net, os.path.join(config.artifacts_path, "model.pth"))

        # Save the model with mlflow
        if self.with_mlflow:
            self.save_mlflow(kwargs['datamodule'])

    def save_mlflow(self, datamodule):
        """
        Save the model with MLFlow.

        Args:
            datamodule (pl.LightningDataModule): a data module with a 'test_dataloader()' method
                used to get a dataset sample.
        """
        # Get a dataset sample with input and output
        dataloader = datamodule.test_dataloader()
        batch = next(itertools.islice(dataloader, 0, None))

        # Build the mlflow signature of the model, from input and output data
        in_data = {k: v for k, v in batch.to_dict().items() if k in ["x",
                                                                     "edge_index",
                                                                     "edge_attr",
                                                                     "u",
                                                                     "batch"]}

        in_data_onnx = in_data # save original data format for ONNX export
        in_data_np = {k: v.detach().numpy() for k, v in in_data.items()}
        out_data_np = {k: v.detach().numpy() for k, v in batch.to_dict().items() if k in ["y", "z"]}

        # signature = infer_signature(
        #     in_data_np,
        #     out_data_np
        # )

        dict_signature = {
            "inputs":
                """[{"name": "x", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3]}},
                    {"name": "edge_index", "type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [2, -1]}},
                    {"name": "edge_attr", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 27]}},
                    {"name": "u", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 17]}},
                    {"name": "batch", "type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]""",
            "outputs":
                """[
                    {"name": "delta_sw_diff", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 138]}},
                    {"name": "delta_sw_add", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 138]}},
                    {"name": "delta_lw_diff", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 138]}},
                    {"name": "delta_lw_add", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 138]}},
                    {"name": "hr_sw", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 137]}},
                    {"name": "hr_lw", "type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 137]}}
                ]"""
        }
        signature = ModelSignature.from_dict(dict_signature)

        _, unconvertible_ops = torch.onnx.utils.unconvertible_ops(self.model, in_data_onnx)
        if unconvertible_ops:
            UserWarning(f"The model uses some onnx ops that are not supported : {unconvertible_ops}. "
                        "Exporting the model to the onnx could fail.")

        # export the model to ONNX generic format
        onnx_model_path = "/tmp/test_onnx.onnx"
        self.model.to_onnx(
            file_path=onnx_model_path,
            input_sample=in_data_onnx,
            input_names=list(in_data_np.keys()),
            output_names=list(out_data_np.keys()),
            dynamic_axes={k: {0: 'batch_size'} if k != "edge_index" else {1: 'batch_size'} for k in in_data_np.keys()}
        )

        # log the ONNX model using MLFlow
        mlflow.onnx.log_model(
            onnx_model=onnx.load(onnx_model_path),
            artifact_path='model_with_signature',
            input_example=in_data_np,
            # registered_model_name=self.model_name,  # automatic save in the model registry
            signature=signature,
            pip_requirements='inference_requirements.txt'
        )


def run(cli: LightningCLI) -> None:
    """
    Run the training script.

    Launch the script with "python3.8 trainer.py --config configs/cnn.yaml".

    Args:
        cli (LightningCLI): the CLI instantiated by the training script.
    """
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == '__main__':
    _cli = LightningCLI(trainer_class=Trainer, run=False)
    if _cli.trainer.with_mlflow:
        mlflow.pytorch.autolog(log_models=False)
        with mlflow.start_run():
            run(_cli)
    else:
        run(_cli)
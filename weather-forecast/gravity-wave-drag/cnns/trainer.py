"""This module proposes Pytorch style a Trainer class for the gwd use-case."""

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
from typing import Iterable, List, Optional, Union

import torch
from lightning import Trainer
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import Logger


class CLITrainer(Trainer):
    """Modified PyTorch Lightning Trainer. It automatically tests, logs, and writes artifacts by the
    end of training.
    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator, None],
        devices: Union[List[int], str, int, None],
        max_epochs: int,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        # TODO: delete.
        # For some reason, those two are mandatory in current version of Lightning.
        fast_dev_run: Union[int, bool] = False,
        callbacks: Union[List[Callback], Callback, None] = None,
    ) -> None:
        """Init the Trainer.

        Args:
            accelerator (Union[str, pl.accelerators.Accelerator, None]): Type of accelerator to use
                for training.
            devices: (Union[List[int], str, int, None]): Devices explicit names to use for training.
            max_epochs (int): Maximum number of epochs if no early stopping logic is implemented.
        """
        self._devices = devices
        super().__init__(
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            # TODO: for some reason, a forward pass happens in the model before datamodule creation.
            num_sanity_val_steps=0,
        )
        self.artifacts_path = os.path.join(self.log_dir, "artifacts")
        self.plots_path = os.path.join(self.log_dir, "plots")
        if self.is_global_zero:
            os.makedirs(self.artifacts_path, exist_ok=True)
            os.makedirs(self.plots_path, exist_ok=True)

    def test(self, **kwargs) -> None:
        """
        Uses superclass test results, but additionally, saves raw results as a JSON file, and stores
        the model weights for future use in inference mode.
        """
        results = super().test(**kwargs)[0]
        if self.is_global_zero:
            with open(os.path.join(self.artifacts_path, "results.json"), "w") as f:
                json.dump(results, f)
            torch.save(self.model, os.path.join(self.artifacts_path, "model.pth"))


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.data_path", "model.init_args.data_path", apply_on="instantiate"
        )


def main():
    """Instantiate the CLI and launch the trainer."""
    cli = MyLightningCLI(
        trainer_class=CLITrainer, parser_kwargs={"parser_mode": "omegaconf"}
    )
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()

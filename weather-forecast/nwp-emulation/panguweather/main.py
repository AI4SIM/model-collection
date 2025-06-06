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
from typing import Any

import torch
from jsonargparse.typing import register_type
from lightning import Trainer
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from callback import GlobalSaveConfigCallback
from utils import slice_deserializer

torch.set_float32_matmul_precision("high")


class CustomTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def log_dir(self) -> str:
        """This function patch the default behaviour of the pytorch trainer.py
        It gets the directory for the current experiment associated with the
        logger.

        Returns:
          dirpath : the path where the logs are stored.
        """
        if len(self.loggers) > 0:
            dirpath = os.getenv("LOGDIR")

            for logger in self.loggers:
                if isinstance(logger, (TensorBoardLogger, CSVLogger)):
                    dirpath = logger.log_dir
                    break
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        assert dirpath is not None
        # needed as mypy considers os.getenv() to return a Optional[str],
        # and log_dir needs to return a str
        return dirpath


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.data_latitude", "model.init_args.latitude", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.data_longitude", "model.init_args.longitude", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.surface_variables",
            "model.init_args.surface_variables",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.plevel_variables",
            "model.init_args.plevel_variables",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.plevels", "model.init_args.plevels", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.constant_masks",
            "model.init_args.constant_masks",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.train_time_step", "model.init_args.time_step", apply_on="instantiate"
        )


def cli_main() -> None:
    MyLightningCLI(
        save_config_kwargs={"overwrite": True},
        save_config_callback=GlobalSaveConfigCallback,
        trainer_class=CustomTrainer,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    register_type(slice, str, slice_deserializer)
    cli_main()

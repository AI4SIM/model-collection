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

import importlib
import logging
import os
import pytorch_lightning as pl
import shutil
from typing import Tuple
import unittest
import yaml

import config


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        with open(os.path.join(config.root_path, 'configs', "default.yaml"), "r") as stream:
            self.params = yaml.safe_load(stream)

    def _extract_classpath(self, path: str) -> Tuple[str]:

        s = path.split(".")
        module = ".".join(s[:-1])
        class_path = s[-1]

        return module, class_path

    def _create_instance(self, params: dict) -> [
            importlib.resources.Package,
            importlib.resources.Resource]:

        m_, class_path = self._extract_classpath(params['class_path'])
        m = importlib.import_module(m_)
        module_class = getattr(m, class_path)
        module = module_class(**params['init_args'])

        return module, module_class

    def _test_datamodule(self) -> None:
        self.datamodule, _ = self._create_instance(self.params['fit']['data'])
        self.assertTrue(isinstance(self.datamodule, pl.LightningDataModule))

    def _test_model(self) -> None:
        self.model, _ = self._create_instance(self.params['fit']['model'])
        self.assertTrue(isinstance(self.model, pl.LightningModule))

    def _test_trainer(self) -> None:
        import trainer

        self.trainer = trainer.Trainer(**self.params['fit']['trainer'])
        self.assertTrue(isinstance(self.trainer, pl.Trainer))

    def test_train(self) -> None:
        self._test_datamodule()
        self._test_model()
        self._test_trainer()

        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        self.trainer.test(model=self.model, datamodule=self.datamodule)

        self.assertTrue(True)

    def tearDown(self) -> None:

        logging.shutdown()
        shutil.rmtree(config.experiment_path, ignore_errors=False)


if __name__ == '__main__':

    pl.utilities.seed.seed_everything(42)
    unittest.main()
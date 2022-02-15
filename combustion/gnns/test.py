"""
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import importlib
import logging
import os
import io
import shutil
from typing import Tuple
import unittest
import pytorch_lightning as pl
import yaml

import config

class TestTraining(unittest.TestCase):
    """
        Testing Class
    """

    def setUp(self) -> None:
        """
            setup
        """
        with io.open(os.path.join(config.root_path, 'configs', "default.yaml"), 
                     "r", encoding='utf8') as stream:
            self.params = yaml.safe_load(stream)

    def _extract_classpath(self, path: str) -> Tuple[str]:
        """
            extract
        """

        split_path = path.split(".")
        module = ".".join(split_path[:-1])
        class_path = split_path[-1]

        return module, class_path

    def _create_instance(self, params: dict) -> [
            importlib.resources.Package,
            importlib.resources.Resource]:

        """
            _create_instance
        """

        m_param, class_path = self._extract_classpath(params['class_path'])
        m_attr = importlib.import_module(m_param)
        module_class = getattr(m_attr, class_path)
        module = module_class(**params['init_args'])

        return module, module_class

    def _test_datamodule(self) -> None:
        """
            _test_datamodule
        """

        self.datamodule, _ = self._create_instance(self.params['fit']['data'])
        self.assertTrue(isinstance(self.datamodule, pl.LightningDataModule))

    def _test_model(self) -> None:
        """
            _test_model
        """

        self.model, _ = self._create_instance(self.params['fit']['model'])
        self.assertTrue(isinstance(self.model, pl.LightningModule))

    def _test_trainer(self) -> None:
        """
            _test_trainer
        """

        import trainer

        self.trainer = trainer.Trainer(**self.params['fit']['trainer'])
        self.assertTrue(isinstance(self.trainer, pl.Trainer))

    def test_train(self) -> None:
        """
            test_train
        """

        self._test_datamodule()
        self._test_model()
        self._test_trainer()

        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        self.trainer.test(model=self.model, datamodule=self.datamodule)

        self.assertTrue(True)

    def tearDown(self) -> None:
        """
            tearDown
        """

        logging.shutdown()
        shutil.rmtree(config.experiment_path, ignore_errors=False)

if __name__ == '__main__':

    pl.utilities.seed.seed_everything(42)
    unittest.main()

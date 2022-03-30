"""
Test trainer file
"""

import unittest
import os
import warnings
import torch

from trainer import Trainer

class TestTrainer(unittest.TestCase):
    """
    trainer test file
    """
    
    def setUp(self) -> None:
        """
        define default parameters
        """

        self.args_cpu = {"max_epochs" : 1, 
                         "accelerator" : "cpu" , 
                         "devices" : [0] 
                        }
        self.args_gpu = {"max_epochs" : 1, 
                         "accelerator" : "gpu" , 
                         "devices" : [0] 
                        }

    def test_trainer(self) -> None:
        """
        test trainer file
        """
        
        if torch.cuda.is_available():
            test_trainer_gpu = Trainer(**self.args_gpu)
        
        # avoids GPU warning when testing CPU usage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            test_trainer_cpu = Trainer(**self.args_cpu)
            self.assertEqual(test_trainer_cpu._devices, None)
        
        
        
if __name__ == '__main__':
    unittest.main()
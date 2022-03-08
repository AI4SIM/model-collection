"""
Test plotter file
"""

import unittest
import plotters
import tempfile
import numpy as np
import os


class TestPlotters(unittest.TestCase):
    """
    plotter test file
    """
    
    def setUp(self) -> None:
        """
        define default parameters
        """
        self.g = 10
        self.f = 3
        
        self.grid = (self.g, self.g, self.g)
        self.model_type = "pytorch"
        self.y = np.stack([1 + np.random.randn(self.g, self.g, self.g) for _ in range(self.f)], axis=0)
        self.y_hat = np.stack([1 + np.random.randn(self.g, self.g, self.g) for _ in range(self.f)], axis=0)
            
        
    def test_dispersion_plot(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            
            tt = plotters.Plotter(self.model_type, self.grid)
            tt = tt.dispersion_plot(self.y, self.y_hat, tempdir)
            
            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"dispersion-plot-{self.model_type}.png")))
            
    def test_histo(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            
            tt = plotters.Plotter(self.model_type, self.grid)
            tt = tt.histo(self.y, self.y_hat, tempdir)
            
            
            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"histogram-{self.model_type}.png")))
            
            
            
    def test_histo2d(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            
            tt = plotters.Plotter(self.model_type, self.grid)
            tt = tt.histo2d(self.y, self.y_hat, tempdir)
            
            
            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"histogram2d-{self.model_type}.png")))
            
            
        
    def test_boxplot(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            
            tt = plotters.Plotter(self.model_type, self.grid)
            tt = tt.boxplot(self.y, self.y_hat, tempdir)
            
            
            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"boxplot-{self.model_type}.png")))
                    
        
        
    def test_total_flame_surface(self):
        
        with tempfile.TemporaryDirectory() as tempdir:
            
            tt = plotters.Plotter(self.model_type, self.grid)
            tt = tt.total_flame_surface(self.y, self.y_hat, tempdir)
                        
            self.assertTrue(len(os.listdir(tempdir)) == self.f)
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"total-flame-surface-{self.model_type}-0.png")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"total-flame-surface-{self.model_type}-1.png")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"total-flame-surface-{self.model_type}-2.png")))
            
            
    def test_cross_section(self):
        
        with tempfile.TemporaryDirectory() as tempdir:

            tt = plotters.Plotter(self.model_type, self.grid)
            tt = tt.cross_section( 1 , self.y, self.y_hat, tempdir)
            
            self.assertTrue(len(os.listdir(tempdir)) == self.f)
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"cross-section-{self.model_type}-0.png")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"cross-section-{self.model_type}-1.png")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, 
                                                        f"cross-section-{self.model_type}-2.png")))
            
            
    
if __name__ == '__main__':
    unittest.main()

"""This module provide a test suite for the plotter.py file."""
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

import unittest
import plotters
import tempfile
import numpy as np
import os


class TestPlotters(unittest.TestCase):
    """Plotter test file."""

    def setUp(self) -> None:
        """Define default parameters."""
        grid = 10
        self.f_num = 3

        self.grid = (grid, grid, grid)
        self.model_type = "pytorch"
        self.y = np.stack(
            [1 + np.random.randn(grid, grid, grid) for _ in range(self.f_num)],
            axis=0
        )
        self.y_hat = np.stack(
            [1 + np.random.randn(grid, grid, grid) for _ in range(self.f_num)],
            axis=0
        )

    def test_dispersion_plot(self):
        """Test the "dispersion_plot" produces the result file."""
        with tempfile.TemporaryDirectory() as tempdir:
            tt = plotters.Plotter(self.model_type, self.grid)
            tt.dispersion_plot(self.y, self.y_hat, tempdir)

            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir,
                                                        f"dispersion-plot-{self.model_type}.png")))

    def test_histo(self):
        """Test the "histo" produces the result file."""
        with tempfile.TemporaryDirectory() as tempdir:
            tt = plotters.Plotter(self.model_type, self.grid)
            tt.histo(self.y, self.y_hat, tempdir)

            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir,
                                                        f"histogram-{self.model_type}.png")))

    def test_histo2d(self):
        """Test the "histo2d" produces the result file."""
        with tempfile.TemporaryDirectory() as tempdir:

            tt = plotters.Plotter(self.model_type, self.grid)
            tt.histo2d(self.y, self.y_hat, tempdir)

            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir,
                                                        f"histogram2d-{self.model_type}.png")))

    def test_boxplot(self):
        """Test the "boxplot" produces the result file."""
        with tempfile.TemporaryDirectory() as tempdir:
            tt = plotters.Plotter(self.model_type, self.grid)
            tt.boxplot(self.y, self.y_hat, tempdir)

            self.assertTrue(len(os.listdir(tempdir)) == 1)
            self.assertTrue(os.path.exists(os.path.join(tempdir,
                                                        f"boxplot-{self.model_type}.png")))

    def test_total_flame_surface(self):
        """Test the "total_flame_surface" produces the result files."""
        with tempfile.TemporaryDirectory() as tempdir:
            tt = plotters.Plotter(self.model_type, self.grid)
            tt.total_flame_surface(self.y, self.y_hat, plot_path=tempdir, save=True)

            self.assertEqual(len(os.listdir(tempdir)), self.f_num)
            self.assertTrue(os.path.exists(os.path.join(
                tempdir,
                f"total-flame-surface-{self.model_type}-0.png"))
            )
            self.assertTrue(os.path.exists(os.path.join(
                tempdir,
                f"total-flame-surface-{self.model_type}-1.png"))
            )
            self.assertTrue(os.path.exists(os.path.join(
                tempdir,
                f"total-flame-surface-{self.model_type}-2.png"))
            )

    def test_cross_section(self):
        """Test the "cross_section" produces the result files."""
        with tempfile.TemporaryDirectory() as tempdir:
            tt = plotters.Plotter(self.model_type, self.grid)
            tt.cross_section(1, self.y, self.y_hat, plot_path=tempdir, save=True)

            self.assertEqual(len(os.listdir(tempdir)), self.f_num)
            self.assertTrue(os.path.exists(os.path.join(tempdir,
                                                        f"cross-section-{self.model_type}-0.png")))
            self.assertTrue(os.path.exists(os.path.join(tempdir,
                                                        f"cross-section-{self.model_type}-1.png")))
            self.assertTrue(os.path.exists(os.path.join(tempdir,
                                                        f"cross-section-{self.model_type}-2.png")))


if __name__ == '__main__':
    unittest.main()

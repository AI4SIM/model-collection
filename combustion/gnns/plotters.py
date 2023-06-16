"""This module proposes some utils features to plot results of the model."""
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

import config

plt.style.use('bmh')
plt.rcParams.update({'font.size': 22})


class Plotter:
    """The Plotter class aims at providing some utils to appreciate the training results.."""

    def __init__(self,
                 model_type: torch.Tensor,
                 grid_shape: torch.Tensor,
                 zslice: int = 16) -> None:
        """Init the Plotter."""
        self.model_type = model_type
        self.grid_shape = grid_shape
        self.zslice = zslice

        self.label_target = r"$\overline{\Sigma}_{target}$"
        self.label_predicted = rf"$\overline{{\Sigma}}_{{{self.model_type}}}$"

    def dispersion_plot(self,
                        y_val: np.ndarray,
                        y_hat: np.ndarray,
                        plot_path=config.plots_path) -> None:
        """Plot the dispersion and save it in image."""
        bins = np.linspace(0, 1250, 10)
        error = np.zeros((bins.shape[0], 2))
        err = np.sqrt((y_val - y_hat)**2)
        for i in range(len(bins) - 1):
            idx = np.logical_and(y_val.flatten() >= bins[i], y_val.flatten() < bins[i + 1])
            if np.all(idx):
                error[i, 0] = err.flatten()[idx].mean()
                error[i, 1] = err.flatten()[idx].std()

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(bins[:-1], error[:-1, 0])
        ax.fill_between(bins[:-1],
                        (error[:-1, 0] - error[:-1, 1]),
                        (error[:-1, 0] + error[:-1, 1]),
                        color='b',
                        alpha=.1)
        ax.set_xlabel(self.label_target)
        ax.set_ylabel(f"$RMSE$({self.label_predicted}, {self.label_target})")
        plt.savefig(os.path.join(plot_path, f"dispersion-plot-{self.model_type}.png"))
        plt.close()

    def histo(self, y_val: np.ndarray, y_hat: np.ndarray, plot_path=config.plots_path) -> None:
        """Plot the histo and save it in image."""
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.hist(y_val.flatten(),
                bins=50,
                density=False,
                histtype="step",
                lw=3,
                label=self.label_target)
        ax.hist(y_hat.flatten(),
                bins=50,
                density=False,
                histtype="step",
                lw=3,
                label=self.label_predicted)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\overline{\Sigma}$")
        # Create new legend handles but use the colors from the existing ones
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

        plt.legend(handles=new_handles, labels=labels)
        plt.savefig(os.path.join(plot_path, f"histogram-{self.model_type}.png"))
        plt.close()

    def histo2d(self, y_val: np.ndarray, y_hat: np.ndarray, plot_path=config.plots_path) -> None:
        """Plot the histo2d and save it in image."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.linspace(0, 1200, 100), np.linspace(0, 1200, 100), 'black')
        histo = ax.hist2d(y_val.flatten(), y_hat.flatten(), bins=100, norm=LogNorm(), cmap='Blues')
        plt.colorbar(histo[3])
        ax.set_xlim(-50, 1200)
        ax.set_ylim(-50, 1200)
        ax.set_xlabel(self.label_target)
        ax.set_ylabel(self.label_predicted)
        plt.savefig(os.path.join(plot_path, f"histogram2d-{self.model_type}.png"))
        plt.close()

    def boxplot(self, y_val: np.ndarray, y_hat: np.ndarray, plot_path=config.plots_path) -> None:
        """Plot the boxplot and save it in image."""
        flat_err = []
        y_val = y_val.reshape((-1,) + self.grid_shape)
        y_hat = y_hat.reshape((-1,) + self.grid_shape)
        for i in range(y_val.shape[0]):
            flat_err.append(np.sqrt((y_val[i, :, :, :] - y_hat[i, :, :, :])**2).flatten())

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.boxplot(flat_err, labels=np.arange(y_val.shape[0]), showmeans=True)
        ax.set_xlabel("Snapshot")
        ax.set_ylabel(f"$RMSE$({self.label_predicted}, {self.label_target})")
        plt.savefig(os.path.join(plot_path, f"boxplot-{self.model_type}"))
        plt.close()

    def total_flame_surface(self,
                            y_target: np.ndarray,
                            y_hat: np.ndarray,
                            y_hat_2: np.ndarray = None,
                            y_hat_3: np.ndarray = None,
                            target_title: str = "Ground Truth",
                            pred_title: str = "Prediction GNN",
                            pred_2_title: str = "Prediction CNN",
                            pred_3_title: str = "Pred",
                            save: bool = False,
                            plot_path=config.plots_path) -> None:
        """Plot the total_flame_surface and save it in image."""
        gt_total_flame_surface = np.stack(y_target, axis=0)
        gt_total_flame_surface = np.sum(gt_total_flame_surface, axis=(2, 3))

        pred_total_flame_surface = np.stack(y_hat, axis=0)
        pred_total_flame_surface = np.sum(pred_total_flame_surface, axis=(2, 3))

        if y_hat_2 is not None:
            pred_total_flame_surface_2 = np.stack(y_hat_2, axis=0)
            pred_total_flame_surface_2 = np.sum(pred_total_flame_surface_2, axis=(2, 3))

        if y_hat_3 is not None:
            pred_total_flame_surface_3 = np.stack(y_hat_3, axis=0)
            pred_total_flame_surface_3 = np.sum(pred_total_flame_surface_3, axis=(2, 3))

        for i in range(gt_total_flame_surface.shape[0]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(gt_total_flame_surface.shape[1]),
                                     y=gt_total_flame_surface[i, :],
                                     mode="lines+markers",
                                     name=target_title))
            fig.add_trace(go.Scatter(x=np.arange(gt_total_flame_surface.shape[1]),
                                     y=pred_total_flame_surface[i, :],
                                     mode="lines+markers",
                                     name=pred_title))
            if y_hat_2 is not None:
                fig.add_trace(go.Scatter(x=np.arange(gt_total_flame_surface.shape[1]),
                                         y=pred_total_flame_surface_2[i, :],
                                         mode="lines+markers",
                                         name=pred_2_title))
            if y_hat_3 is not None:
                fig.add_trace(go.Scatter(x=np.arange(gt_total_flame_surface.shape[1]),
                                         y=pred_total_flame_surface_3[i, :],
                                         mode="lines+markers",
                                         name=pred_3_title))
            fig.update_layout(width=1280, height=640)
            fig.update_xaxes(title_text="x position")
            fig.update_yaxes(title_text="Total flame surface")

            # fig.show()
            if save:
                fig.write_image(os.path.join(plot_path,
                                             f"total-flame-surface-{self.model_type}-{i}.png"))

    def cross_section(self,
                      zslice: int,
                      y_val: np.ndarray,
                      y_hat: np.ndarray,
                      y_title: str = 'Ground Truth',
                      y_hat_title: str = 'Prediction',
                      norm_val: float = 1,
                      save: bool = False,
                      plot_path=config.plots_path) -> None:
        """Plot the cross_section and save it in image."""
        for i in range(y_hat.shape[0]):
            y_val = y_val.reshape((-1,) + self.grid_shape)
            y_hat = y_hat.reshape((-1,) + self.grid_shape)

            sigma = y_val[i, :, :, :]
            sigma = np.moveaxis(sigma, [0, -1], [-1, 0])

            prediction = y_hat[i, :, :, :]
            prediction = np.moveaxis(prediction, [0, -1], [-1, 0])

            fig = make_subplots(rows=1, cols=6,
                                subplot_titles=[y_title, y_hat_title, "Difference"],
                                specs=[[{}, None, {}, None, {}, None]])
            fig.add_trace(go.Contour(z=sigma[zslice, :, :],
                                     zmin=0,
                                     zmax=950 / norm_val,
                                     contours=dict(showlines=False,
                                                   showlabels=False,),
                                     line=dict(width=0),
                                     contours_coloring='heatmap',
                                     colorscale="IceFire",
                                     colorbar=dict(x=0.27)), row=1, col=1)
            fig.add_trace(go.Contour(z=prediction[zslice, :, :],
                                     zmin=0,
                                     zmax=950 / norm_val,
                                     contours=dict(showlines=False,
                                                   showlabels=False,),
                                     line=dict(width=0),
                                     contours_coloring='heatmap',
                                     colorscale="IceFire",
                                     colorbar=dict(x=0.62)), row=1, col=3)
            fig.add_trace(go.Contour(z=sigma[zslice, :, :] - prediction[zslice, :, :],
                                     zmin=-300 / norm_val,
                                     zmax=300 / norm_val,
                                     contours=dict(showlines=False,
                                                   showlabels=False,),
                                     line=dict(width=0),
                                     contours_coloring='heatmap',
                                     colorscale="RdBu_r",
                                     colorbar=dict(x=0.97)), row=1, col=5)

            fig.update_layout(width=1756, height=450)
            fig.update_xaxes(title_text="x direction", row=1, col=1)
            fig.update_xaxes(title_text="x direction", row=1, col=3)
            fig.update_xaxes(title_text="x direction", row=1, col=5)
            fig.update_yaxes(title_text="y direction", row=1, col=1)
            fig.update_yaxes(title_text="y direction", row=1, col=3)
            fig.update_yaxes(title_text="y direction", row=1, col=5)
            fig.update_layout(xaxis=dict(domain=[0, 0.27]),
                              xaxis2=dict(domain=[0.35, 0.62]),
                              xaxis3=dict(domain=[0.7, 0.97])
                              )

            # fig.show()
            if save:
                fig.write_image(os.path.join(plot_path, f"cross-section-{self.model_type}-{i}.png"))

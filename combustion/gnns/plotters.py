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
import data

plt.style.use('bmh')
plt.rcParams.update({'font.size': 22})

class Plotter:
    
    def __init__(self, 
                 y: torch.Tensor, 
                 y_hat: torch.Tensor, 
                 model_type: torch.Tensor, 
                 grid_shape: torch.Tensor, 
                 zslice: int = 16) -> None:
        
        self.y = y
        self.y_hat = y_hat
        self.model_type = model_type
        self.grid_shape = grid_shape
        self.zslice = zslice
        
        self.label_target = "$\overline{\Sigma}_{target}$"
        self.label_predicted = "$\overline{{\Sigma}}_{{{}}}$".format(self.model_type)
        
        self.dispersion_plot()
        self.histo()
        self.histo2d()
        self.boxplot()
        
        dataset = data.CombustionDataset(config.data_path)
        self.total_flame_surface(dataset)
        self.cross_section(self.zslice, dataset)
    
    def dispersion_plot(self) -> None:
        
        bins = np.linspace(0, 1250, 10)
        error = np.zeros((bins.shape[0], 2))
        err = np.sqrt((self.y - self.y_hat)**2)
        for i in range(len(bins)-1):
            idx = np.logical_and(self.y.flatten() >= bins[i], self.y.flatten() < bins[i+1])
            error[i, 0] = err.flatten()[idx].mean()
            error[i, 1] = err.flatten()[idx].std()
            
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(bins[:-1], error[:-1, 0])
        ax.fill_between(bins[:-1], (error[:-1, 0]-error[:-1, 1]), (error[:-1, 0]+error[:-1, 1]), color='b', alpha=.1)
        ax.set_xlabel(self.label_target)
        ax.set_ylabel("$RMSE$({}, {})".format(self.label_predicted, self.label_target))
        plt.savefig(os.path.join(config.plots_path, "dispersion-plot-{}.png".format(self.model_type)))
        plt.close()
        
    def histo(self) -> None:
        
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.hist(self.y.flatten(), bins=50, density=False, histtype="step", lw=3, label=self.label_target)
        ax.hist(self.y_hat.flatten(), bins=50, density=False, histtype="step", lw=3, label=self.label_predicted)
        ax.set_yscale("log")
        ax.set_xlabel("$\overline{\Sigma}$")
        # Create new legend handles but use the colors from the existing ones
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]

        plt.legend(handles=new_handles, labels=labels)
        plt.savefig(os.path.join(config.plots_path, "histogram-{}.png".format(self.model_type)))
        plt.close()
        
    def histo2d(self) -> None:
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.linspace(0, 1200, 100), np.linspace(0, 1200, 100), 'black')
        h = ax.hist2d(self.y.flatten(), self.y_hat.flatten(), bins=100, norm=LogNorm(), cmap='Blues');
        plt.colorbar(h[3])
        ax.set_xlim(-50, 1200)
        ax.set_ylim(-50, 1200)
        ax.set_xlabel(self.label_target)
        ax.set_ylabel(self.label_predicted)
        plt.savefig(os.path.join(config.plots_path, "histogram2d-{}.png".format(self.model_type)))
        plt.close()
        
    def boxplot(self) -> None:
        
        flat_err = []
        self.y = self.y.reshape((-1,) + self.grid_shape)
        self.y_hat = self.y_hat.reshape((-1,) + self.grid_shape)
        for i in range(self.y.shape[0]):
            flat_err.append(np.sqrt((self.y[i, :, :, :] -self.y_hat[i, :, :, :])**2).flatten())
        
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.boxplot(flat_err, labels=np.arange(self.y.shape[0]), showmeans=True)
        ax.set_xlabel("Snapshot")
        ax.set_ylabel("$RMSE$({}, {})".format(self.label_predicted, self.label_target))
        plt.savefig(os.path.join(config.plots_path, "boxplot-{}".format(self.model_type)))
        plt.close()
    
    def total_flame_surface(self, 
                            cd_test: torch.Tensor) -> None:
        
        gt_total_flame_surface = np.stack(self.y, axis=0)
        gt_total_flame_surface = np.sum(gt_total_flame_surface, axis=(2, 3))
        
        pred_total_flame_surface = np.stack(self.y_hat, axis=0)
        pred_total_flame_surface = np.sum(pred_total_flame_surface, axis=(2, 3))
        
        for i in range(gt_total_flame_surface.shape[0]):
            dns3_time = cd_test.raw_file_names[i].split("_")[-1].split(".")[0]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(gt_total_flame_surface.shape[1]),
                                     y=gt_total_flame_surface[i, :],
                                     mode="lines+markers",
                                     name="DNS"))
            fig.add_trace(go.Scatter(x=np.arange(gt_total_flame_surface.shape[1]),
                                     y=pred_total_flame_surface[i, :],
                                     mode="lines+markers",
                                     name=self.model_type))
            fig.update_layout(title_text="Time : {}".format(dns3_time), width=1280, height=640)
            fig.update_xaxes(title_text="x position")
            fig.update_yaxes(title_text="Total flame surface")

            fig.write_image(os.path.join(config.plots_path, "total-flame-surface-{}-{}.png".format(self.model_type, dns3_time)))
    
    def cross_section(self, zslice: int, cd_test: torch.Tensor) -> None:
        for i in range(self.y_hat.shape[0]):
            dns3_time = cd_test.raw_file_names[i].split("_")[-1].split(".")[0]
            print("Time : {}".format(dns3_time))

            sigma = self.y[i, :, :, :]
            sigma = np.moveaxis(sigma, [0, -1], [-1, 0])

            prediction = self.y_hat[i, :, :, :]
            prediction = np.moveaxis(prediction, [0, -1], [-1, 0])

            fig = make_subplots(rows=1, cols=6,
                                subplot_titles=["Ground Truth", "{}".format(self.model_type), "Difference"],
                                specs=[[{}, None, {}, None, {}, None]])
            fig.add_trace(go.Contour(z=sigma[zslice, :, :],
                                  zmin=0,
                                  zmax=950,
                                  contours=dict(showlines=False,
                                                showlabels=False,),
                                  line=dict(width=0),
                                  contours_coloring='heatmap',
                                  colorscale="IceFire",
                                  colorbar = dict(x=0.27)), row=1, col=1)
            fig.add_trace(go.Contour(z=prediction[zslice, :, :],
                                  zmin=0,
                                  zmax=950,
                                  contours=dict(showlines=False,
                                                showlabels=False,),
                                  line=dict(width=0),
                                  contours_coloring='heatmap',
                                  colorscale="IceFire",
                                  colorbar = dict(x=0.62)), row=1, col=3)
            fig.add_trace(go.Contour(z=sigma[zslice, :, :]-prediction[zslice, :, :],
                                  zmin=-300,
                                  zmax=300,
                                  contours=dict(showlines=False,
                                                showlabels=False,),
                                  line=dict(width=0), 
                                  contours_coloring='heatmap',
                                  colorscale="RdBu_r",
                                  colorbar=dict(x=0.97)), row=1, col=5)

            fig.update_layout(title_text="Time : {}".format(dns3_time), width=1756, height=450)
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
            fig.write_image(os.path.join(config.plots_path, "cross-section-{}-{}.png".format(self.model_type, dns3_time)))
        
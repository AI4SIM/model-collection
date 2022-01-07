import config
import io
import numpy as np
import torch
import plotly.graph_objects as go

def flame_surface(batch_idx, y_hat, y, stage: str):

    def _flame_surface(tensor):
        return torch.sum(tensor, axis=(-2, -1)).cpu()

    y_hat_total_flame_surface = _flame_surface(y_hat)
    y_total_flame_surface = _flame_surface(y)
    x = np.arange(len(y_total_flame_surface))

    fig = go.Figure(layout={'width': 1200, 'height': 600})
    fig.add_trace(go.Scatter(x=x, y=y_hat_total_flame_surface, mode='lines', name='y_hat'))
    fig.add_trace(go.Scatter(x=x, y=y_total_flame_surface, mode='lines', name='y'))

    return fig
# import config
import plotly
import plotly.graph_objects as go


def sigma_slices(y_hat, y, slice_idx=6):
    
    y_hat_slice = y_hat[:, :, slice_idx].cpu().numpy().T
    
    y_slice = y[:, :, slice_idx].cpu().numpy().T
    
        
    fig = plotly.subplots.make_subplots(rows=1, cols=6, subplot_titles=["Ground Truth", "GNN", "Δ"],
                        specs=[[{}, None, {}, None, {}, None]])
        
        
    fig.add_trace(go.Contour(
        z=y_slice,
        zmin=0,
        zmax=950,
        contours=dict(showlines=False, showlabels=False,),
        line=dict(width=0),
        contours_coloring='heatmap',
        colorscale="IceFire",
        colorbar=dict(x=0.27)), 
        row=1, 
        col=1)
    fig.add_trace(go.Contour(
        z=y_hat_slice,
        zmin=0,
        zmax=950,
        contours=dict(showlines=False, showlabels=False,),
        line=dict(width=0),
        contours_coloring='heatmap',
        colorscale="IceFire",
        colorbar=dict(x=0.62)), 
        row=1, 
        col=3)
    fig.add_trace(go.Contour(
        z=y_slice - y_hat_slice,
        zmin=-300,
        zmax=300,
        contours=dict(showlines=False, showlabels=False,),
        line=dict(width=0),
        contours_coloring='heatmap',
        colorscale="RdBu_r",
        colorbar=dict(x=0.97)), 
        row=1, 
        col=5)

    fig.update_layout(title_text="", width=1756, height=450)
    fig.update_xaxes(title_text="x direction", row=1, col=1)
    fig.update_xaxes(title_text="x direction", row=1, col=3)
    fig.update_xaxes(title_text="x direction", row=1, col=5)
    fig.update_yaxes(title_text="y direction", row=1, col=1)
    fig.update_yaxes(title_text="y direction", row=1, col=3)
    fig.update_yaxes(title_text="y direction", row=1, col=5)
    fig.update_layout(
        xaxis=dict(domain=[0, 0.27]),
        xaxis2=dict(domain=[0.35, 0.62]),
        xaxis3=dict(domain=[0.7, 0.97]))

    return fig
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
from scipy.spatial import ConvexHull,  Delaunay
from scipy.interpolate import griddata
import sys

def plot_capacity_vs_cost(ds1,ds2,ds3,ds4):
    fig = make_subplots(rows=4,cols=4,shared_xaxes=False,shared_yaxes=True,
                subplot_titles=('OCGT',
                                'wind',
                                'solar',
                                'transmission'),
                horizontal_spacing = 0.05,
                vertical_spacing= 0.05)


    datasets = [ds1,ds2,ds3,ds4]

    for k in range(4):
        for i in range(4):

            ds = datasets[k]
            x = ds.interrior_points_cost
            y = ds.interrior_points[:,i]
            y = y*1e-3


            # Calculate quantiles
            q_0 = []
            q_08 = []
            q_341 = []
            q_5 = []
            q_659 = []
            q_92 = []
            q_1 = []
            q_x = []

            for j in [0,1,2,5,10]:
                filter = x <= min(x)*(1+j/100)
                q_x.append(j/100)

                q_0.append(np.quantile(y[filter],0.0))
                q_08.append(np.quantile(y[filter],0.08))
                q_341.append(np.quantile(y[filter],0.341))
                q_5.append(np.quantile(y[filter],0.5))
                q_659.append(np.quantile(y[filter],0.659))
                q_92.append(np.quantile(y[filter],0.92))
                q_1.append(np.quantile(y[filter],1))
            # Plot of points
            #fig.add_trace(go.Scatter(x=x/min(x)-1,y=y,mode='markers'))
            # Plot of 100% and 0 % quantiles
            fig.add_trace(go.Scatter(x=np.concatenate([np.flip(q_x),q_x]),
                                    y=np.concatenate([np.flip(q_1),q_0]),
                                    fill='toself',
                                    fillcolor='#1f77b4',
                                    mode='lines+markers',
                                    marker=dict(color='#1f77b4'),
                                    line=dict(shape='spline',smoothing=1)),row=k+1,col=i+1)
            # Plot of 2 sigma quantiles
            fig.add_trace(go.Scatter(x=np.concatenate([np.flip(q_x),q_x]),
                                    y=np.concatenate([np.flip(q_92),q_08]),fill='toself',
                                    mode='lines+markers',
                                    marker=dict(color='#ff7f0e'),
                                    fillcolor='#ff7f0e',
                                    line=dict(shape='spline',smoothing=1)),row=k+1,col=i+1)
            # Plot of 1 sigma quantiles
            fig.add_trace(go.Scatter(x=np.concatenate([np.flip(q_x),q_x]),
                                    y=np.concatenate([np.flip(q_659),q_341]),fill='toself',
                                    mode='lines+markers',
                                    marker=dict(color='#2ca02c'),
                                    fillcolor='#2ca02c',
                                    line=dict(shape='spline',smoothing=1)),row=k+1,col=i+1)
            # Plot of 50% quantile
            fig.add_trace(go.Scatter(x=q_x,y=q_5,marker=dict(color='#7f7f7f')),row=k+1,col=i+1)


            fig.update_xaxes(title_text="MGA slack", row=4, col=i+1)
            CO2_values = [0,50,80,95]
            fig.update_yaxes(title_text="{} % CO2 reduction <br> Installed capacity [GW]".format(CO2_values[k]), row=k+1, col=1,range=[0,2.3e3])

    fig.update_xaxes(showline=True, 
                    linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    zerolinecolor="LightGray",
                    zerolinewidth=1,
                    #range=[0,100],
                    )
    fig.update_yaxes(showline=True, 
                    linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    zerolinecolor="LightGray",
                    zerolinewidth=1,
                    #range=[0,1],
                    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=1000,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig



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


def plot_histogram(ds1,ds2,ds3,ds4):

    fig = make_subplots(rows=5,cols=1,shared_xaxes=True,shared_yaxes=True,
                row_heights=[0.25,0.25,0.25,0.25,0.0001],
                subplot_titles=('Buisness as usual',
                                '50% CO2 reduction',
                                '80% CO2 reduction',
                                '95% CO2 reduction',
                                ''),
                        vertical_spacing= 0.05)

    fig1 = ds1.plot_histogram()
    fig2 = ds2.plot_histogram()
    fig3 = ds3.plot_histogram()
    fig4 = ds4.plot_histogram()



    fig.add_traces(fig1.data[:],rows=[1]*len(fig1.data),cols=[1]*len(fig1.data))
    fig.add_traces(fig2.data[:],rows=[2]*len(fig2.data),cols=[1]*len(fig2.data))
    fig.add_traces(fig3.data[:],rows=[3]*len(fig3.data),cols=[1]*len(fig3.data))
    fig.add_traces(fig4.data[:],rows=[4]*len(fig4.data),cols=[1]*len(fig4.data))

    fig.update_xaxes(title_text="Installed capacity [GW]",showticklabels=True, row=4, col=1)

    fig.update_xaxes(showline=True, 
                    linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    zerolinecolor="LightGray",
                    zerolinewidth=1,
                    #range=[0,100],
                    )
    fig.update_yaxes(showline=True,
                    showticklabels=True,
                    title_text='Probability Density',
                    linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    zerolinecolor="LightGray",
                    zerolinewidth=1,
                    range=[0,0.0065],
                    )


    # Legend
    fig.add_trace(go.Scatter(x=[30,30,30,30],y=[3,2,1,4],
                            mode='markers',
                            xaxis='x2',
                            yaxis='y2',
                            marker=dict(size=10,
                                        color=['#1f77b4','#ff7f0e','#2ca02c','#8c564b'],)), 
                            row=5, col=1)
    fig.add_trace(go.Scatter(x=[2e2,2e2,2e2,2e2],y=[1,2,3,4],
                            mode='text',
                            xaxis='x2',
                            yaxis='y2',
                            text=['Transmission','Solar','Wind','OCGT',],
                            textposition="middle right" ),
                            row=5, col=1)
    # Legend
    fig.update_xaxes(domain=[0.85,1],
                    #range=[-1,1],
                    showline=False,
                    showticklabels=False,
                    showgrid=False,
                    zerolinecolor='rgba(0,0,0,0)',
                    linewidth=0,
                    zerolinewidth=0,  
                    row=5,col=1)
    fig.update_yaxes(domain=[0.85,1],
                    range=[0,5],
                    showline=False,
                    showticklabels=False,
                    showgrid=False,
                    zerolinecolor='rgba(0,0,0,0)',
                    linewidth=0,
                    zerolinewidth=0, 
                    title_text='',
                    row=5,col=1)


    fig.update_layout(
        autosize=False,
        width=800,
        height=1000,
        showlegend=False,
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,1)',
    )

    return fig

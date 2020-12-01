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

def plot_optimal_solutions_energy(ds1,ds2,ds3,ds4):
    datasets = [ds1,ds2,ds3,ds4]

    data = []
    for ds in datasets:
        data.append([ds.df_generation['wind g'][0],
                    ds.df_generation['solar g'][0],
                    ds.df_generation['ocgt g'][0],
                    ])

    data = np.array(data)

    fig = go.Figure()

    names = ['wind','solar','ocgt','transmission']
    colors = ['#1f77b4','#ff7f0e','#8c564b','#2ca02c']

    for i in range(3):
        fig.add_trace(go.Scatter(
                        name=names[i],
                        x=[0,50,80,95],
                        y=data[:,i]*1e-6,
                        line=dict(color=colors[i])
                        #yaxis='y'+str(i+1)
        ))

    fig.update_layout(
        autosize=False,
        width=700,
        height=500,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showline=True, linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    range=[0,100],
                    title_text='% CO2 reduction')
    fig.update_yaxes(showline=True, 
                    linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    #range=[0,1],
                    title_text='Produced energy [TWh]')
    return fig 


def plot_optimal_solutions_power(ds1,ds2,ds3,ds4):
    data = []
    datasets =  [ds1,ds2,ds3,ds4]

    for ds in datasets:
        data.append([ds.df_points['wind'][0],
                    ds.df_points['solar'][0],
                    ds.df_points['ocgt'][0],
                    ds.df_detail['transmission'][0],
                    ds.df_detail['co2_emission'][0],
                    ds.df_detail['objective_value'][0],
                    ds.df_detail['gini'][0]   ])

    data = np.array(data)

    fig = go.Figure()

    names = ['wind','solar','ocgt','transmission']
    colors = ['#1f77b4','#ff7f0e','#8c564b','#2ca02c']

    for i in range(4):
        fig.add_trace(go.Scatter(
                        name=names[i],
                        x=[0,50,80,95],
                        y=data[:,i]*1e-3,
                        line=dict(color=colors[i])
                        #yaxis='y'+str(i+1)
        ))

    fig.update_layout(
        autosize=False,
        width=700,
        height=500,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showline=True, linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    range=[0,100],
                    title_text='% CO2 reduction')
    fig.update_yaxes(showline=True, 
                    linewidth=1, 
                    linecolor='black',
                    gridcolor="LightGray",
                    #range=[0,1],
                    title_text='Installed capacity [GW]')

    return fig

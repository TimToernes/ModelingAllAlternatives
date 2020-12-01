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


def plot_correlation(datasets):

    ds = datasets[-1]
    ocgt = np.empty([1,1])
    wind = np.empty([1,1])
    solar = np.empty([1,1])
    transmission = np.empty([1,1])
    for ds in datasets:

        ocgt = np.append(ocgt,ds.interrior_points[:,0])
        wind = np.append(wind,ds.interrior_points[:,1])
        solar = np.append(solar,ds.interrior_points[:,2])
        transmission = np.append(transmission,ds.interrior_points[:,3])
    #gini = ds.interrior_points_gini

    plot_range = [0,2e3]

    variables= [ocgt,wind,solar,transmission]
    labels= ['ocgt','wind','solar','transmission']
    domains = [[0,0.24],[0.26,0.49],[0.51,0.74],[0.76,1]]
    colors = ['#8c564b','#1f77b4','#ff7f0e','#2ca02c']

    corr_df = pd.DataFrame(data=np.array(variables).T,columns=['ocgt','wind','solar','transmission'])
    corr_matrix = corr_df.corr()

    fig = make_subplots(rows = 4, cols=4,shared_xaxes=False,shared_yaxes=False)

    variables_r = np.array(variables)[:,np.random.rand(len(ocgt))>0.9]


    for i in range(4):
        for j in range(4):
            
            if i != j and j>=i:
                fig.add_trace(go.Scatter(y=[0],
                                            x=[0],
                                            mode='text',
                                            text='{:.2f}'.format(corr_matrix.iloc[i][j]),
                                            textfont_size=20,
                                            yaxis='y2',xaxis='x2'),row=i+1,col=j+1)
                fig.update_yaxes(range=[-1,1],
                                showticklabels=False,
                                showline=False,
                                showgrid=False,
                                row=i+1,col=j+1)
                fig.update_xaxes(range=[-1,1],
                                showticklabels=False,
                                showline=False,
                                showgrid=False,
                                row=i+1,col=j+1)

            if i != j and i>=j:
                # Plot scatter
                fig.add_trace(go.Scatter(y=variables_r[i]*1e-3,
                                            x=variables_r[j]*1e-3,
                                            mode='markers',
                                            marker=dict(opacity=0.25,color='#17becf'),
                                            yaxis='y2',xaxis='x2'),row=i+1,col=j+1)
                fig.update_yaxes(range=plot_range,
                                domain=domains[3-i],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)
                fig.update_xaxes(range=plot_range,
                                domain=domains[j],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)

            elif i == j :
                # Plot histogram
                fig.add_trace(go.Histogram(x=variables[i]*1e-3,
                                            xaxis='x',
                                            histnorm='probability',
                                            marker_color=colors[i]),
                                            row=i+1,col=j+1)
                fig.update_xaxes(range=plot_range,
                                domain=domains[j],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)
                fig.update_yaxes(
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                domain=domains[3-i],
                                row=i+1,col=j+1)
            # Set axis titles for border plots
            if i == 3 :
                fig.update_xaxes(title_text=labels[j],showticklabels=True,
                                domain=domains[j],
                                row=i+1,col=j+1)

            if j == 0 :
                fig.update_yaxes(title_text=labels[i],showticklabels=True,
                                domain=domains[3-i],
                                row=i+1,col=j+1)
                if i == 0:
                    fig.update_yaxes(showticklabels=False,row=i+1,col=j+1)


    fig.update_layout(width=1000,
                        height=1000,
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showline=True,linecolor='black'),
                        yaxis=dict(showline=True,linecolor='black'),
                        )
    return fig 



import pypsa 
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

def import_network(path):
    network = pypsa.Network()
    network.import_from_hdf5(path)
    network.snapshots = network.snapshots[0:2]
    return network

#network = import_network('data/networks/euro_30')

#%%

def plot_network(network_path,ds,datasets, titles=['(a)','b)','(c)','(d)']):

    network = import_network(network_path)

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=2,
        #column_widths=[0.6, 0.4],
        row_heights=[0.45,0.45,0.15],
        subplot_titles=['','']+titles,
        specs=[[{"type": 'Scattergeo'},{"type": 'Scattergeo'}],
               [ {"type": 'Scattergeo'},{"type": 'Scattergeo'}],
              [ {"type": 'Scatter'},{"type": 'Scatter'}]])


    for i in range(4):

        #network.generators.p_nom_opt=datasets[i].df_detail.iloc[0,0:111]
        #network.links.p_nom_opt=datasets[i].df_detail.iloc[0,222:-4]
        network.generators.p_nom_opt= ds.df_detail.iloc[datasets[i]]
        network.links.p_nom_opt     =ds.df_links.iloc[datasets[i]]
        network.storage_units.p_nom_opt = ds.df_store_p.iloc[datasets[i]]



        # Links
        import matplotlib.cm as cm
        for link in network.links.iterrows():

            bus0 = network.buses.loc[link[1]['bus0']]
            bus1 = network.buses.loc[link[1]['bus1']]
            cap = link[1]['p_nom_opt']
            cap_max = max(network.links.p_nom_opt)

            fig.add_trace(go.Scattergeo(
                locationmode = 'country names',
                geo = 'geo'+str(i+1),
                lon = [bus0.x,bus1.x],
                lat = [bus0.y,bus1.y],
                mode = 'lines',
                line = dict(width = cap/cap_max*5+0.5,color = 'green'),
                ),row=int(np.floor(i/2)+1),col=i%2+1)



        # Bar plots 
        for bus in network.buses.iterrows():

            filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='wind')]
            wind = sum(network.generators[filter].p_nom_opt)
            filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='solar')]
            solar = sum(network.generators[filter].p_nom_opt)
            filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='ocgt')]
            ocgt = sum(network.generators[filter].p_nom_opt)
            filter = [x and y for x,y in zip(network.storage_units.bus==bus[0],network.storage_units.carrier=='H2')]
            H2 = sum(network.storage_units[filter].p_nom_opt)
            filter = [x and y for x,y in zip(network.storage_units.bus==bus[0],network.storage_units.carrier=='battery')]
            battery = sum(network.storage_units[filter].p_nom_opt)

            fig.add_trace(go.Scattergeo(
            locationmode = 'country names',
            lon = [bus[1]['x']-1.5,bus[1]['x']-0.75,bus[1]['x'],bus[1]['x']+0.75,bus[1]['x']+1.5 ],
            lat = [bus[1]['y'], bus[1]['y'],bus[1]['y'],bus[1]['y'],bus[1]['y']],
            geo = 'geo'+str(i+1),
            mode = 'markers',
            marker = dict(
                size = np.array([wind,solar,ocgt,H2,battery])/4000,
                symbol = 'line-ns',
                opacity=0.95,
                line = dict(
                    width = 10,
                    color = ['#1f77b4','#ff7f0e','#8c564b','#e377c2','#d62728'],
                ),
            )),row=int(np.floor(i/2)+1),col=i%2+1)


    """ Colors
        '#1f77b4',  // muted blue
        '#ff7f0e',  // safety orange
        '#2ca02c',  // cooked asparagus green
        '#d62728',  // brick red
        '#9467bd',  // muted purple
        '#8c564b',  // chestnut brown
        '#e377c2',  // raspberry yogurt pink
        '#7f7f7f',  // middle gray
        '#bcbd22',  // curry yellow-green
        '#17becf'   // blue-teal
    """
    #Legend
    # Capacity
    fig.add_trace(go.Scatter(
        x = [0.1,0.13,0.16,0.2,0.23,0.26,0.3,0.33,0.36,0.4,0.43,0.46,0.5,0.53,0.56],
        y = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
        mode = 'markers',
        marker = dict(
            size = [25,12.5,5,25,12.5,5,25,12.5,5,25,12.5,5,25,12.5,5],
            symbol = 'line-ns',
            opacity=0.95,
            line = dict(
                width = 7,
                color = ['#1f77b4','#1f77b4','#1f77b4',
                        '#ff7f0e','#ff7f0e','#ff7f0e',
                        '#8c564b','#8c564b','#8c564b',
                        '#e377c2','#e377c2','#e377c2',
                        '#d62728','#d62728','#d62728'],
            )),
    ),row=3,col=1)

    # Transmission
    fig.add_trace(go.Scatter(
        x = [0.6,0.63,0.66,],
        y = [6,6,6],
        mode = 'markers',
        marker = dict(
            size = 15,
            symbol = 'line-ew',
            opacity=0.95,
            line = dict(
                width = [50*1e3/cap_max*5+0.5,25*1e3/cap_max*5+0.5,0.5],
                color = ['green','green','green'],
            )),
    ),row=3,col=1)
    # Text
    fig.add_trace(go.Scatter(
        x = [0.13,0.23,0.33,0.44,0.54,0.64,0.1,0.13,0.16,0.2,0.23,0.26,0.3,0.33,0.36,0.4,0.43,0.46,0.5,0.53,0.56,0.6,0.63,0.66,],
        y = [9,9,9,9,9,9,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
        text = ['Wind [GW]','Solar [GW]','Backup [GW]','H2 [GW]','Battery [GW]','Transmission [GW]','100','50','20','100','50','20','100','50','20','100','50','20','100','50','20','50','25','0'],
        mode = 'text',
        textposition="middle center"
    ),row=3,col=1)

    fig.update_geos(
            scope = 'europe',
            projection_type = 'azimuthal equal area',
            showland = True,
            landcolor = 'rgb(203, 203, 203)',
            countrycolor = 'rgb(204, 204, 204)',
            showocean=False,
            #domain=dict(x=[0,1],y=[0,1]),
            lataxis = dict(
                range = [35, 64],
                showgrid = False
            ),
            lonaxis = dict(
                range = [-11, 26],
                showgrid = False
        ))
    fig.update_layout(
        geo2=dict(
            domain=dict(x=[0.52,0.99],y=[0.55,1])),#Top Rigth
        geo1=dict(
            domain=dict(x=[0,0.48],y=[0.55,1])),#Top Left
        geo3=dict(
            domain=dict(x=[0,0.48],y=[0.1,0.55])),#Botom left
        geo4=dict(
            domain=dict(x=[0.52,0.99],y=[0.1,0.55])),#Botom right
    )
    fig.update_traces( textfont_size=14)
    fig.update_layout(
        autosize=False,
        showlegend=False,
        uniformtext_minsize=14,
        xaxis=dict(showticklabels=False,visible=False,range=[0.05,0.75],domain=[0,1]),
        yaxis=dict(showticklabels=False,visible=False,range=[0,10],domain=[0,0.08]),
            paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1000,
        height=1000,
        margin=dict(l=5, r=5, t=5, b=5,pad=0),
        )

    return fig

def plot_topology(network_path,ds_co2_95):

    network = import_network(network_path)


 
        
    fig = go.Figure()

    # Nodes
    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = network.buses.x,
        lat = network.buses.y,
        hoverinfo = 'text',
        text = network.buses.index,
        mode = 'markers',
        marker = dict(
            size = 5,
            color = 'black',
            line = dict(
                width = 3,
                color = 'rgba(68, 68, 68, 0)'
            )
        )))

    fig.add_trace(go.Scattergeo(lon=[min(network.buses.x),max(network.buses.x)],
                                lat=[np.mean(network.buses.y),np.mean(network.buses.y)],
                                mode='lines'
                                ))

    # Links
    import matplotlib.cm as cm
    for link in network.links.iterrows():

        bus0 = network.buses.loc[link[1]['bus0']]
        bus1 = network.buses.loc[link[1]['bus1']]
        cap = link[1]['p_nom_opt']

        fig.add_trace(go.Scattergeo(
            locationmode = 'country names',
            lon = [bus0.x,bus1.x],
            lat = [bus0.y,bus1.y],
            mode = 'lines',
            line = dict(width = cap/2e-13,color = 'green'),
            ))

    # Bar plots 
    network.generators.p_nom_opt=ds_co2_95.df_detail.iloc[0,0:111]


    for bus in network.buses.iterrows():

        filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='wind')]
        wind = sum(network.generators[filter].p_nom_opt)
        filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='solar')]
        solar = sum(network.generators[filter].p_nom_opt)
        filter = [x and y for x,y in zip(network.generators.bus==bus[0],network.generators.type=='ocgt')]
        ocgt = sum(network.generators[filter].p_nom_opt)

        fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = [bus[1]['x'],bus[1]['x']+0.5,bus[1]['x']-0.5 ],
        lat = [bus[1]['y'], bus[1]['y'],bus[1]['y']],
        hoverinfo = 'text',
        text = bus[0],
        mode = 'markers',
        marker = dict(
            size = np.array([wind,solar,ocgt])/2000,
            color = ['blue','yellow','black'],
            symbol = 'line-ns',
            line = dict(
                width = 10,
                color = ['blue','yellow','black'],
            )
        )))

    # Legend 
    fig.add_trace(go.Scattergeo(
    locationmode = 'country names',
    lon = [-11,-11,-11],
    lat = [62,63,64],
    hoverinfo = 'text',
    text = 'legend',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = ['blue','yellow','black'],
        symbol = 'line-ns',
        line = dict(
            width = 10,
            color = ['blue','yellow','black'],
        )
    )))



    fig.update_layout(
        title_text = 'Euro-30 model',
        showlegend = False,
        geo = go.layout.Geo(
            scope = 'europe',
            projection_type = 'azimuthal equal area',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
            lataxis = dict(
                range = [35, 64],
                showgrid = False
            ),
            lonaxis = dict(
                range = [-11, 26],
                showgrid = False
            )
        ),
    )

    return fig


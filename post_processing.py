# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython
#from IPython.display import display, clear_output
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
#import pypsa_tools as pt
from pypsa_tools import *
import iso3166
#import logging
im_dir="C:/Users\Tim\Dropbox\MGA paper\Paper/figures/"

#df_detail = pd.read_csv('./output/df_local_Scandinavia_co2_eta_0.1.csv')


###################################################################################
#%% ########################## Data processing ####################################
###################################################################################


ds_local_2D = datasPet('./output/local_euro_00_2D_eta_0.1.csv')

ds_co2_00 = dataset(['./output/prime_euro_00_4D_eta_0.1.csv',
                    './output/prime_euro_00_4D_eta_0.05.csv',
                    './output/prime_euro_00_4D_eta_0.02.csv',
                    './output/prime_euro_00_4D_eta_0.01.csv'])

ds_co2_50 = dataset(['./output/prime_euro_50_4D_eta_0.1.csv',
                    './output/prime_euro_50_4D_eta_0.05.csv',
                    './output/prime_euro_50_4D_eta_0.02.csv',
                    './output/prime_euro_50_4D_eta_0.01.csv'])

ds_co2_80 = dataset(['./output/prime_euro_80_4D_eta_0.1.csv',
                    './output/prime_euro_80_4D_eta_0.05.csv',
                    './output/prime_euro_80_4D_eta_0.02.csv',
                    './output/prime_euro_80_4D_eta_0.01.csv'])

ds_co2_95 = dataset(['./output/prime_euro_95_4D_eta_0.1.csv',
                    './output/prime_euro_95_4D_eta_0.05.csv',
                    './output/prime_euro_95_4D_eta_0.02.csv',
                    './output/prime_euro_95_4D_eta_0.01.csv'])


ds_all = dataset(['./output/prime_euro_00_4D_eta_0.1.csv',
                    './output/prime_euro_00_4D_eta_0.05.csv',
                    './output/prime_euro_00_4D_eta_0.02.csv',
                    './output/prime_euro_00_4D_eta_0.01.csv',
                    './output/prime_euro_50_4D_eta_0.1.csv',
                    './output/prime_euro_50_4D_eta_0.05.csv',
                    './output/prime_euro_50_4D_eta_0.02.csv',
                    './output/prime_euro_50_4D_eta_0.01.csv',
                    './output/prime_euro_80_4D_eta_0.1.csv',
                    './output/prime_euro_80_4D_eta_0.05.csv',
                    './output/prime_euro_80_4D_eta_0.02.csv',
                    './output/prime_euro_80_4D_eta_0.01.csv',
                    './output/prime_euro_95_4D_eta_0.1.csv',
                    './output/prime_euro_95_4D_eta_0.05.csv',
                    './output/prime_euro_95_4D_eta_0.02.csv',
                    './output/prime_euro_95_4D_eta_0.01.csv']
                    )

ds_all_05 = dataset([
                    './output/prime_euro_00_4D_eta_0.05.csv',
                    './output/prime_euro_50_4D_eta_0.05.csv',
                    './output/prime_euro_80_4D_eta_0.05.csv',
                    './output/prime_euro_95_4D_eta_0.05.csv']
                    )

ds_all_01 = dataset([
                    './output/prime_euro_00_4D_eta_0.01.csv',
                    './output/prime_euro_50_4D_eta_0.01.csv',
                    './output/prime_euro_80_4D_eta_0.01.csv',
                    './output/prime_euro_95_4D_eta_0.01.csv']
                    )

#%% plot histogram

fig = plot_histogram(ds_co2_00,
                ds_local_2D,
                ds_co2_80,
                ds_co2_95)
#fig.write_image(im_dir+"4D_study_histogram.pdf")
fig.show()

#%% Plot of capacity vs cost

fig = plot_capacity_vs_cost(ds_co2_00,
                ds_co2_50,
                ds_co2_80,
                ds_co2_95)
#fig.write_image(im_dir+"Capacaty_vs_cost.pdf")
fig.show()

#%% Plot of optimal solutions sumary - Capacity

fig = plot_optimal_solutions_power(ds_co2_00,
            ds_co2_50,
            ds_co2_80,
            ds_co2_95)
#fig.write_image(im_dir+"optimal_solutions_summary.pdf")

fig.show()
#%% Plot of optimal solutions sumary - Production

fig = plot_optimal_solutions_energy(ds_co2_00,
                ds_co2_50,
                ds_co2_80,
                ds_co2_95)
#fig.write_image(im_dir+"optimal_solutions_summary_production.pdf")
fig.show()

#%% Network plots 

fig = plot_network('data/networks/euro_30',
                    [ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95])
#fig.write_image(im_dir+"Optimal_solutions.pdf")
fig.show()

#Plot of topology
plot_topology('data/networks/euro_30',ds_co2_95)
fig.show()

#%% Corelation plots

#datasets=[ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95]
datasets=[ds_co2_95]
fig = plot_correlation(datasets)
#fig.write_image(im_dir+"Corelation_4D_95.pdf")
fig.show()
#%%

datasets=[ds_co2_95]

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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

vegetables = corr_matrix.index
farmers = corr_matrix.index

harvest = corr_matrix.values.T


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, "{0:1.2f}".format(harvest[i, j]),
                       ha="center", va="center", color="w")

#ax.set_title("Harvest of local farmers (in tons/year)")
#fig.tight_layout()
plt.show()

#%% Gini plot based on energy production 

datasets = [ds_all_01,ds_all_05,ds_all]
names = ['1% slack', '5% slack', '10% slack',]

x_data = []
y_data = []
for ds in datasets:
    co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100
    try :
        x_data.append(np.concatenate([x_data[-1],co2_reduction]))
        y_data.append(np.concatenate([y_data[-1],ds.interrior_points_gini]))
    except : 
        x_data.append(co2_reduction)
        y_data.append(ds.interrior_points_gini)


fig = plot_gini(x_data,y_data,x_title='Energy production gini coefficient')
fig.show()
fig.write_image(im_dir+"energy_production_gini.pdf")

#%% GINI on cost

network = pypsaTools.import_network('data/networks/euro_30')

def calc_gini_capex(ds,network):
    
    gini_list = []

    buses = network.buses.index
    techs = ['onwind','offwind','solar','ocgt']
    

    capex = dict(onwind=1040000, 
                solar=300000, 
                ocgt=430000,
                offwind=1930000)

    for i in range(ds.df_detail.shape[0]):
        row = ds.df_detail.iloc[i,0:111]
        expenses = dict.fromkeys(buses,0)

        for bus in buses:
            for tech in techs:
                try:
                    expenses[bus] = expenses[bus] + row[bus+' '+tech]*capex[tech]
                except : 
                    pass
    

        network.buses.total_expenses = list(expenses.values())

        
        # Add network total load info to network.buses
        load_total= [sum(network.loads_t.p_set[load]) for load in network.loads_t.p_set.columns]
        network.buses['total_load']=load_total


        rel_demand = network.buses.total_load/sum(network.buses.total_load)
        rel_generation = network.buses.total_expenses/sum(network.buses.total_expenses)
        
        # Rearange demand and generation to be of increasing magnitude
        idy = np.argsort(rel_generation/rel_demand)
        rel_demand = rel_demand[idy]
        rel_generation = rel_generation[idy]


        # Calculate cumulative sum and add [0,0 as point
        rel_demand = np.cumsum(rel_demand)
        rel_demand = np.concatenate([[0],rel_demand])
        rel_generation = np.cumsum(rel_generation)
        rel_generation = np.concatenate([[0],rel_generation])

        lorenz_integral= 0

        for i in range(len(rel_demand)-1):
            lorenz_integral += (rel_demand[i+1]-rel_demand[i])*(rel_generation[i+1]-rel_generation[i])/2 + (rel_demand[i+1]-rel_demand[i])*rel_generation[i]
            
        gini = 1- 2*lorenz_integral

        gini_list.append(gini)

    try :
        ds.df_detail = ds.df_detail.join(pd.DataFrame(dict(gini_capex=gini_list)))
    except :
        ds.df_detail['gini_capex'] = gini_list

    ds.interrior_points_gini_capex = griddata(ds.df_points.values, 
                                    ds.df_detail['gini_capex'], 
                                    ds.interrior_points, 
                                    method='linear') 


    return ds


ds_all = calc_gini_capex(ds_all,network)
ds_all_05 = calc_gini_capex(ds_all_05,network)
ds_all_01 = calc_gini_capex(ds_all_01,network)


datasets = [ds_all_01,ds_all_05,ds_all]
names = ['1% slack', '5% slack', '10% slack',]

x_data = []
y_data = []
for ds in datasets:
    co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100
    x_data.append(co2_reduction)
    y_data.append(ds.interrior_points_gini_capex)


fig = plot_gini(x_data,y_data,x_title='CPAEX gini coefficient')
fig.show()
fig.write_image(im_dir+"CAPEX_gini.pdf")

#%% plot on land use 

network = pypsaTools.import_network('data/networks/euro_30')

def calc_land_use(ds,network):

    buses = network.buses.index
    techs = ['onwind','offwind','solar','ocgt']
    land_use_list = []


    land_fill = dict(onwind=10, 
                solar=145, 
                ocgt=0,
                offwind=0)

    for i in range(ds.df_detail.shape[0]):
        row = ds.df_detail.iloc[i,111:222]
        land_use = dict.fromkeys(buses,0)

        for bus in buses:
            for tech in techs:
                try:
                    land_use[bus] = land_use[bus] + row[bus+' '+tech+' g']*land_fill[tech]
                except : 
                    pass


        land_use_list.append(sum(land_use.values()))

    try :
        ds.df_detail = ds.df_detail.join(pd.DataFrame(dict(land_use=land_use_list)))
    except :
        ds.df_detail['land_use'] = land_use_list

    ds.interrior_points_land_use = griddata(ds.df_points.values, 
                                    ds.df_detail['land_use'], 
                                    ds.interrior_points, 
                                    method='linear') 

    return ds
  

ds_all = calc_land_use(ds_all,network)
ds_all_05 = calc_land_use(ds_all_05,network)
ds_all_01 = calc_land_use(ds_all_01,network)


datasets = [ds_all_01,ds_all_05,ds_all]
names = ['1% slack', '5% slack', '10% slack',]

x_data = []
y_data = []
for ds in datasets:
    co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100
    try :
        x_data.append(np.concatenate([x_data[-1],co2_reduction]))
        y_data.append(np.concatenate([y_data[-1],ds.interrior_points_land_use]))
    except :
        x_data.append(co2_reduction)
        y_data.append(ds.interrior_points_land_use)


fig = plot_gini(x_data,y_data,x_title='Land use [km2]')
fig.show()
fig.write_image(im_dir+"Land_use.pdf")


#%% Implementation time

def calc_implementation_time(ds,network):
    buses = network.buses.index
    techs = ['onwind','offwind','solar','ocgt']
    # data source: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
    df_gdp = pd.read_csv('data/GDP_USD.csv',header=2)
    df_gdp.index = df_gdp['Country Code']
    usd_to_eur = 0.92
    gdp_dict = {bus:df_gdp.at[iso3166.countries_by_alpha2[bus].alpha3,'2007']*usd_to_eur for bus in buses}
    max_use_of_gdp = 0.1 # % of GDP that can be used on energy investement

    implementation_time_list = []

    capex = dict(onwind=1040000, 
                solar=300000, 
                ocgt=430000,
                offwind=1930000)

    for i in range(ds.df_detail.shape[0]):
        row = ds.df_detail.iloc[i,0:111]
        expenses = dict.fromkeys(buses,0)

        for bus in buses:
            for tech in techs:
                try:
                    expenses[bus] = expenses[bus] + row[bus+' '+tech]*capex[tech]
                except : 
                    pass
        
        implementation_time = {bus:expenses[bus]/(gdp_dict[bus]*max_use_of_gdp) for bus in buses}

        implementation_time_list.append(max(implementation_time.values()))

    try :
        ds.df_detail = ds.df_detail.join(pd.DataFrame(dict(implementation_time=implementation_time_list)))
    except :
        ds.df_detail['implementation_time'] = implementation_time_list

    ds.interrior_points_implementation = griddata(ds.df_points.values, 
                                    ds.df_detail['implementation_time'], 
                                    ds.interrior_points, 
                                    method='linear') 

    return ds

ds_all = calc_implementation_time(ds_all,network)
ds_all_05 = calc_implementation_time(ds_all_05,network)
ds_all_01 = calc_implementation_time(ds_all_01,network)


datasets = [ds_all_01,ds_all_05,ds_all]
names = ['1% slack', '5% slack', '10% slack',]

x_data = []
y_data = []
for ds in datasets:
    co2_reduction = (1-ds.interrior_points_co2/max(ds.interrior_points_co2)) *100
    try :
        x_data.append(np.concatenate([x_data[-1],co2_reduction]))
        y_data.append(np.concatenate([y_data[-1],ds.interrior_points_implementation]))
    except :
        x_data.append(co2_reduction)
        y_data.append(ds.interrior_points_implementation)


fig = plot_gini(x_data,y_data,x_title='Implementation time [years]')
fig.show()
fig.write_image(im_dir+"Implementation_time.pdf")




#%% CO2 vs wind/solar mix

fig = go.Figure()
co2_reduction = (1-ds_all.interrior_points_co2/max(ds_all.interrior_points_co2)) *100
#mix = ds_all.interrior_points[:,1]/(ds_all.interrior_points[:,2]+ds_all.interrior_points[:,1])
mix = ds_all.interrior_points_gini_capex


fig.add_trace(go.Scatter(x=mix,
                        y = co2_reduction,
                        mode='markers' ,
                        name='10% slack'))


co2_reduction = (1-ds_all_01.interrior_points_co2/max(ds_all.interrior_points_co2)) *100
#mix = ds_all_01.interrior_points[:,1]/(ds_all_01.interrior_points[:,2]+ds_all_01.interrior_points[:,1])
mix = ds_all_01.interrior_points_gini_capex

fig.add_trace(go.Scatter(x=mix,
                        y = co2_reduction,
                        mode='markers',
                        name='1%slack' ))


fig.update_yaxes(title_text='CO2 emission reduction [%]',showgrid=False)
fig.update_xaxes(title_text='$\\alpha$',showgrid=False)

fig.update_layout(width=800,
                    height=500,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showline=True,linecolor='black'),
                    yaxis=dict(showline=True,linecolor='black'),
                    )
#fig.write_image(im_dir+"Corelation_mix_co2.pdf")
fig.show()

#%% Table data 

ds=[ds_co2_00,ds_co2_50,ds_co2_80,ds_co2_95]
techs = ['wind','solar','ocgt']


for tech in techs:
    d1 = ds[0].df_points[tech][0]*1e-3
    d2 = ds[1].df_points[tech][0]*1e-3
    d3 = ds[2].df_points[tech][0]*1e-3
    d4 = ds[3].df_points[tech][0]*1e-3

    print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format(tech,d1,d2,d3,d4))

techs2 = ['transmission','gini','co2_emission','objective_value']
factor = [1e-3,1,1e-6,1e-9]

for tech,f in zip(techs2,factor):
    d1 = ds[0].df_detail[tech][0]*f
    d2 = ds[1].df_detail[tech][0]*f
    d3 = ds[2].df_detail[tech][0]*f
    d4 = ds[3].df_detail[tech][0]*f   

    print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\'.format(tech,d1,d2,d3,d4))



#%% Title page figure 

fig = plot_titlefig()
fig.show()





#%% Test section ########################################################
##########################################################################

#ds_local_2D.df_points
x_data= ds_local_2D.df_points['wind']
y_data = ds_local_2D.df_points['solar']

fig = go.Figure()
gini_hull = ConvexHull(np.array([x_data,y_data]).T)

x = gini_hull.points[gini_hull.vertices][:,1]
y = gini_hull.points[gini_hull.vertices][:,0]

fig.add_trace(go.Scatter(x=np.append(x,x[0]),
                    y=np.append(y,y[0]),
                    mode='lines',
                    fill='tonexty'
                    ))
#%%
self = ds_co2_50

hull = self.hull
df_points = self.df_points
co2_emission = self.co2_emission
objective_value = self.objective_value
interrior_points = self.interrior_points
interrior_points_cost = self.interrior_points_cost
interrior_points_co2 = griddata(df_points.values, co2_emission, interrior_points, method='linear')

#%% CO2 vs price 
fig = make_subplots(rows = 1, cols=1)


fig.add_trace(go.Scatter(y=interrior_points_cost,
                            x=self.interrior_points_gini,
                            mode='markers'),row=1,col=1)

fig.add_trace(go.Scatter(y=self.df_detail['objective_value'],
                            x=self.df_detail['gini'],
                            mode='markers',marker={'size':10}),row=1,col=1)

fig.update_yaxes(title_text='co2 emission [ton/year]')
fig.update_xaxes(title_text='cost [€]')


#%%
fig = go.Figure()

fig.add_trace(go.Histogram(x=self.interrior_points_gini,
                                        marker_color=colors[i]))


#%%
import networkx as nx
import matplotlib.pyplot as plt

df = pd.DataFrame({'wind':wind,'ocgt':ocgt,'solar':solar,'transmission':transmission})
corr = df.corr()
# Transform it in a links data frame (3 columns only):
links = corr.stack().reset_index()
links.columns = ['var1', 'var2','value']
links
 
# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
links_filtered=links.loc[  (links['var1'] != links['var2']) ]
links_filtered
 
# Build your graph
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
 
# Plot the network:
#nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)

pos = nx.spring_layout(G,weight='weight')
nx.draw(G,pos=pos, width=2, with_labels=True)
#%% Step by step

#interrior_points_co2 = griddata(df_points.values, co2_emission, interrior_points, method='linear')
df_points = ds_co2_95.df_points
cost = ds_co2_95.objective_value
interrior_points = ds_co2_95.interrior_points
trace1 = (go.Scatter(x=df_points['ocgt'][0:1],
                            y=df_points['wind'][0:1],
                            #z=df_points['solar'][0:1],
                            mode='markers',
                            marker={'color':'blue'}))

trace2 = (go.Scatter(x=df_points['ocgt'][1:5],
                            y=df_points['wind'][1:5],
                            #z=df_points['solar'][1:],
                            mode='markers',
                            marker={'color':cost,'colorbar':{'thickness':20,'title':'Scenario cost'}}))
"""
trace3 = (go.Scatter(x=interrior_points[:,0],
                                    y=interrior_points[:,1],
                                    #z=interrior_points[:,2],
                                    mode='markers',
                                    marker={'size':2,'color':'pink'}))#,
                                            #'color':self.interrior_points_cost,
                                            #'colorbar':{'thickness':20,'title':'Scenario cost'}}))
"""
# Points generated randomly

fig = go.Figure(layout={'width':900,
                        'height':800,
                        'showlegend':False},
                data=[trace2,trace1])

#ds_co2_80.hull = ConvexHull(df_points[['ocgt','wind']][0:],qhull_options='C-1e3')#,qhull_options='Qj')#,qhull_options='C-1')#,qhull_options='A-0.999')
#ds_co2_80=ds_co2_80.create_interior_points()
#ds_co2_80=ds_co2_80.calc_interrior_points_cost()

# Plot of hull facets
"""
points = hull.points
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    fig.add_trace(go.Mesh3d(x=points[s, 0], 
                                y=points[s, 1], 
                                z=points[s, 2],
                                opacity=0.2,
                                color='aquamarine'
                                ))
# Plot of vectors

for i in range(len(hull.equations)):
    fig.add_trace(go.Cone(  x=[np.mean(hull.points[hull.simplices[i]],axis=0)[0]], 
                            y=[np.mean(hull.points[hull.simplices[i]],axis=0)[1]], 
                            z=[np.mean(hull.points[hull.simplices[i]],axis=0)[2]], 
                            u=[hull.equations[i,0]*100000], 
                            v=[hull.equations[i,1]*100000], 
                            w=[hull.equations[i,2]*100000],
                            showscale=False))
"""
fig.update_layout(scene = dict(
                    xaxis_title='ocgt',
                    yaxis_title='wind',
                    camera=dict(eye=dict(x=-1.25,y=1.25,z=1.25))))

fig.show()


















# %%

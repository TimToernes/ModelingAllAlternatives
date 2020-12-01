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
from scipy.interpolate import griddata,interpn
import sys
import copy
#import pypsa_tools as pt
from pypsa_tools import *
import iso3166
#import logging
im_dir="C:/Users\Tim\Dropbox\MGA paper\Paper/figures/"



#%%

ds = dataset(['output/prime_euro_95_storage_5D_eta_0.3.xlsx',
            'output/prime_euro_95_storage_5D_eta_0.15.xlsx', ],
                variables=['wind','solar','H2','battery','co2_emission','system_cost'],
                m = 500000)


#ds = dataset(['output/prime_euro_80_storage_5D_eta_0.05.xlsx','output/prime_euro_80_storage_5D_eta_0.1.xlsx'],
#                variables=['wind','solar','H2','battery','co2_emission'])



#ds = dataset('output/archive/prime_euro_80_4D_eta_0.1.csv',
#            data_type='csv',
#            variables=['ocgt','wind','solar','transmission'])

#%% Figure 1 - Topology


case_scenarios = [0,                                        # Optimum
                ds.df_secondary_metrics.gini.idxmin(),      # low gini
                ds.df_sum_var.wind.idxmax(),                # high wind
                ds.df_secondary_metrics.co2_emission.idxmin()] # low co2 emission
print(case_scenarios)
fig = plot_network('data/networks/euro_80_storage',
                    ds,case_scenarios,['<b>a</b> - Optimum','<b>b</b> - High equality','<b>c</b> - Large wind capacity','<b>d</b> - Low CO2 emission'])
fig.show()

fig.write_image('figure1.pdf')



# %% Figure 3 - Correlation/histogram

def contour(x,y):

    gini_hull = ConvexHull(np.array([x,y]).T)

    x = gini_hull.points[gini_hull.vertices][:,0]
    y = gini_hull.points[gini_hull.vertices][:,1]

    trace = go.Scatter(x=np.append(x,x[0]),
                        y=np.append(y,y[0]),
                        mode='lines',
                        #name=names[i],
                        fill='tonexty'
                        )
    return trace

def calc_interrior(ds):
    variables = ds.variables
    ds.interrior_points = np.concatenate([ds.interrior_points,ds.df_sum_var[ds.variables].values])
    ds.df_interrior = pd.DataFrame(data=ds.interrior_points,
                                    columns=variables)


    def interpolate(ds,key):
        try : 
            ds.df_interrior[key] = griddata(ds.df_sum_var[ds.variables],
                                    ds.df_secondary_metrics[key],
                                    ds.interrior_points, 
                                    method='linear',
                                    rescale=True,
                                    )
        except Exception as e:
            print(e)
            print(var)

        return ds 


    variables = ['gini','autoarky','system_cost','co2_reduction','land_use','gini_capex','implementation_time']



    ds.df_interrior['transmission'] =  griddata(ds.df_sum_var[ds.variables],
            ds.df_sum_var['transmission'],
            ds.interrior_points, 
            method='linear') 

    ds.df_interrior['ocgt'] =  griddata(ds.df_sum_var[ds.variables],
            ds.df_sum_var['ocgt'],
            ds.interrior_points, 
            method='linear') 
            
    for var in variables : 
        ds = interpolate(ds,var)

    ds.df_interrior = ds.df_interrior.dropna() 

    """

    key= 'gini'
    ds.df_interrior[key] = griddata(ds.df_sum_var[ds.variables],
                                    ds.df_secondary_metrics[key],
                                    ds.interrior_points, 
                                    method='linear',
                                   rescale=True,
                                    )
    key= 'autoarky'
    ds.df_interrior[key] = griddata(ds.df_sum_var[ds.variables],
                                    ds.df_secondary_metrics[key],
                                    ds.interrior_points, 
                                    method='linear',
                                   rescale=True,
                                    )
    key= 'system_cost'
    ds.df_interrior[key] = griddata(ds.df_sum_var[ds.variables],
                                    ds.df_secondary_metrics[key],
                                    ds.interrior_points, 
                                    method='linear',
                                   rescale=True,
                                    )

    try :
            key= 'co2_reduction'
            ds.df_interrior[key] = griddata(ds.df_sum_var[ds.variables],
                                    ds.df_secondary_metrics[key],
                                    ds.interrior_points, 
                                    method='linear',
                                   rescale=True,
                                    )
    except : 
        pass 
    """

    #for key in ds.df_secondary_metrics.keys():
    #    ds.df_interrior[key] = griddata(ds.df_sum_var[ds.variables],
    #                                ds.df_secondary_metrics[key],
    #                                ds.interrior_points, 
    #                                method='linear',
    #                               rescale=True,
    #                                )

    

    return ds 

def plot_contour(x,y,plot_range=None):

    if plot_range == None:
        [[min(x),max(x)],[min(y),max(y)]]

    h,x_bins,y_bins = np.histogram2d(x,y,bins=30,
                                range=plot_range,
                                )

    trace = go.Contour(x=x_bins,y=y_bins,z=h.T,
    line_smoothing=1.3,
    showscale = False,
    contours=dict(start=1,end=751,size=250),
    colorscale = [[0, 'rgba(0,0,255,0)'],[0.5, 'rgba(0,0,255,0.4)'],[1.0, 'rgba(0,0,255,0.9)']]
    )

    return trace

def data_processing(df):

    #df[['wind','solar','H2','battery']] = df[['wind','solar','H2','battery']]*1e-3
    #df['co2_emission'] = 100 - (df['co2_emission']/571067122.7405636)*100  #1151991000.0
    df['co2_emission'] = 100 - (df['co2_emission']/1510e6)*100  #1151991000.0
    #df['system_cost'] = df['system_cost']/ds.df_secondary_metrics['system_cost'][0]*100-100
    for tech in ['ocgt','wind','solar','H2','battery']:
        try :
            df[tech] = df[tech]*1e-3
        except:
            pass
    #df['co2_emission'] = 100 - (df['co2_emission']/571067122.7405636)*100  #1151991000.0
    df['system_cost'] = df['system_cost']/ds.df_secondary_metrics['system_cost'][0]*100-100
    #df['system_cost'] = df['system_cost']/164620377000.0*100 -100
    return df

def figure3(ds,scenario_idx,show_max_min_points=False):

    #variables=ds.variables
    variables = ['wind', 'solar', 'H2', 'battery']
    ds = calc_interrior(ds)
    # Scaling of variables
    df = ds.df_interrior[['system_cost','co2_emission','gini','ocgt'] + variables]
    #df = ds.df_interrior[['system_cost', 'co2_emission', 'gini', 'wind', 'solar', 'H2', 'battery']]
    keys = df.keys()


    df_2 = ds.df_sum_var[variables]
    df_2['ocgt'] = ds.df_sum_var['ocgt']
    df_2['gini'] = ds.df_secondary_metrics['gini']
    df_2['co2_emission'] = ds.df_secondary_metrics['co2_emission']
    df_2['system_cost'] = ds.df_secondary_metrics['system_cost']


    df_2 = data_processing(df_2)
    df_2 = df_2[df.keys()]
    df = data_processing(df)

    data_shape = df.shape[1]

    corr_matrix = df.corr()
    fig = make_subplots(rows = data_shape, cols=data_shape,shared_xaxes=False,shared_yaxes=False)

    #plot_range = [0,1e3]
    #domains = [[0,0.24],[0.26,0.49],[0.51,0.74],[0.76,1]]
    spacing = 0.01
    domains = list(zip(np.linspace(-spacing,1,data_shape+1)[:-1]+spacing,np.linspace(0,1,data_shape+1)[1:]-spacing))
    #domains = domains[-data_shape-1:]
    #colors = ['#0e58a8','#30a4ca','#9be4ef','#f3d573','#da8200','#ac2301','#820000','#820000']
    colors = ['#155764','#1C7D7D','#239780','#2BB174','#34CB5F','#3EE544','#6DFF48','#B9FF85','#ECFFC6']
    #colors = ['#8A9B0F','#F8CA00','#E97F02','#BD1550','#490A3D','#CDB380','#036564','#BB575C','#78446C','#30307B']

    label_dic = {'H2': 'Hydrogen storage <br> [GW]',
                'wind': 'Wind turbines <br> [GW]',
                'solar': 'Solar PV <br> [GW]',
                'battery': 'Battery <br> [GW]',
                'gini': 'Self-sufficiency <br> Gini coefficient',
                'system_cost': 'Cost increase <br> [%]',
                'co2_emission': 'CO<sub>2</sub> reduction <br> [%]',
                'ocgt': 'Backup capacity <br> [GW]',
                'transmission': 'Transmission <br> [GW]'}


    labels = [label_dic[key] for key in df.keys()]

    data = df.values.T
    data_optimum = df_2.values.T
    data_ranges = np.array([df_2.min().values,df_2.max().values]).T

    diff = (data_ranges[:,1]-data_ranges[:,0])*0.04
    data_ranges[:,0] = data_ranges[:,0]-diff
    data_ranges[:,1] = data_ranges[:,1]+diff


    for i in range(data_shape):
        for j in range(data_shape):
            
            if i != j and j>=i:
                # Write correlation value on left side of matrix
                fig.add_trace(go.Scatter(y=[0],
                                            x=[0],
                                            mode='text',
                                            text='{:.2f}'.format(corr_matrix.iloc[i][j]),
                                            textfont_size=20,
                                            yaxis='y2',xaxis='x2'),row=i+1,col=j+1)
                fig.update_yaxes(range=[-1,1],
                                domain=domains[data_shape-1-i],
                                showticklabels=False,
                                showline=False,
                                showgrid=False,
                                row=i+1,col=j+1)
                fig.update_xaxes(range=[-1,1],
                                domain=domains[j],
                                showticklabels=False,
                                showline=False,
                                showgrid=False,
                                row=i+1,col=j+1)

            if i != j and i>=j:
                # Plot scatter
                for idx,color,symbol in zip(scenario_idx,['red','gray','black','black'],['circle','asterisk','x-thin','hash']):
                    fig.add_trace(go.Scatter(x=[data_optimum[j][idx]],y=[data_optimum[i][idx]],
                                                marker=dict(color=color,size=12,opacity=1,
                                                            symbol=symbol ,
                                                            line=dict(width=2,color='orange'))
                                                            ),row=i+1,col=j+1)
                if show_max_min_points:
                    fig.add_trace(go.Scatter(x=data_optimum[j][0:12],
                                            y=data_optimum[i][0:12],
                                            mode='markers',
                                            marker=dict(color='red'),
                                            ),row=i+1,col=j+1)

                #fig.add_trace(go.Scatter(x=data[j][0:1000],y=data[i][0:1000],mode='markers'),row=i+1,col=j+1)
                fig.add_trace(plot_contour(x=data[j],
                                            y=data[i],
                                            plot_range=[data_ranges[j],data_ranges[i]]),
                                            row=i+1,col=j+1)

                
                fig.update_yaxes(range=data_ranges[i],
                                domain=domains[data_shape-1-i],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)
                fig.update_xaxes(range=data_ranges[j],
                                domain=domains[j],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)

            elif i == j :
                # Plot histogram on diagonal
                fig.add_trace(go.Histogram(x=data[i],
                                            xaxis='x',
                                            histnorm='probability',
                                            nbinsx=50,
                                            marker_color=colors[i]),
                                            row=i+1,col=j+1)
                fig.update_xaxes(range=data_ranges[i],
                                domain=domains[j],
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                row=i+1,col=j+1)
                fig.update_yaxes(
                                showticklabels=False,
                                showline=True,
                                linecolor='black',
                                domain=domains[data_shape-1-i],
                                row=i+1,col=j+1)
            # Set axis titles for border plots
            if i == data_shape-1 :
                fig.update_xaxes(#title_text=labels[j],
                                showticklabels=True,
                                tickangle =45,
                                domain=domains[j],
                                row=i+1,col=j+1)
                fig.add_annotation(dict(
                                        y=-0.08,
                                        #x=[0,0.15 ,0.34 , 0.5, 0.63 , 0.84 ,0.96,1][j],
                                        x=[0,0.125 ,0.26 , 0.43, 0.57 , 0.73 ,0.88,0.98][j],
                                        showarrow=False,
                                        align="center",
                                        text=labels[j],
                                        textangle=0,
                                        xref="paper",
                                        yref="paper"))

            if j == 0 :
                fig.update_yaxes(showticklabels=True,
                                domain=domains[data_shape-1-i],
                                tickangle=-45,
                                row=i+1,col=j+1)
                fig.add_annotation(dict(
                                    x=-0.07,
                                    #y=[0.05,0.16 ,0.34 , 0.5, 0.635 , 0.86 ,0.98,1][-1-i],
                                    y=[0.01,0.12 ,0.27 , 0.43, 0.56 , 0.74 ,0.87,0.99][-1-i],
                                    showarrow=False,
                                    text=labels[i],
                                    textangle=-90,
                                    align="center",
                                    xref="paper",
                                    yref="paper"))  
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
    fig.show()
    return fig

#%%

fig3 = figure3(copy.deepcopy(ds),case_scenarios)
fig3.write_image('figure3.pdf')

#%%

fig5 = figure3(copy.deepcopy(ds),[0],True)
fig5.write_image('figure5.pdf')




#%% Figure 3 legend

fig = go.Figure()


fig.add_trace(go.Scatter(x=[1],y=[0],text=['Optimum'],mode='markers+text',textposition='middle right',
                        marker=dict(color='red',size=12,
                                        line=dict(width=2,color='orange'))))
#fig.add_trace(go.Scatter(x=[1],y=[0],text=['High equality'],mode='markers+text',textposition='middle right',
#                        marker=dict(color='orange',size=12,symbol='asterisk',
#                                        line=dict(width=2,color='orange'))))
#
#fig.add_trace(go.Scatter(x=[2],y=[0],text=['Low CO <sub>2 </sub> emission'],mode='markers+text',textposition='middle right',
#                        marker=dict(color='orange',size=12,symbol='hash',
#                                        line=dict(width=2,color='orange'))))
#
#fig.add_trace(go.Scatter(x=[3],y=[0],text=['Large wind capacity'],mode='markers+text',textposition='middle right',
#                        marker=dict(color='orange',size=12,symbol='x-thin',
#                                        line=dict(width=2,color='orange'))))
#
fig.add_trace(go.Scatter(x=[2],y=[0],text=['max-min MGA scenario'],mode='markers+text',textposition='middle right',
                        marker=dict(color='red',)))

fig.update_layout(      width = 1000,
                        height = 200,
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(visible=False,range=[-0.1,4]),
                        yaxis=dict(visible=False),
                        )
fig.show()
fig.write_image('figure5_legend.pdf')
# %% CSV files
""" 
ds_all_01 = dataset([
                    './output/archive/prime_euro_00_4D_eta_0.01.csv',
                    './output/archive/prime_euro_50_4D_eta_0.01.csv',
                    './output/archive/prime_euro_80_4D_eta_0.01.csv',
                    './output/archive/prime_euro_95_4D_eta_0.01.csv'],
                    data_type='csv',
                    variables=['ocgt','wind' ,'solar','transmission']
                    )

ds_all_05 = dataset([
                    './output/archive/prime_euro_00_4D_eta_0.05.csv',
                    './output/archive/prime_euro_50_4D_eta_0.05.csv',
                    './output/archive/prime_euro_80_4D_eta_0.05.csv',
                    './output/archive/prime_euro_95_4D_eta_0.05.csv'],
                    data_type='csv',
                    variables=['ocgt','wind' ,'solar','transmission']
                    )

ds_all_10 = dataset([
                    './output/archive/prime_euro_00_4D_eta_0.1.csv',
                    './output/archive/prime_euro_50_4D_eta_0.1.csv',
                    './output/archive/prime_euro_80_4D_eta_0.1.csv',
                    './output/archive/prime_euro_95_4D_eta_0.1.csv'],
                    data_type='csv',
                    variables=['ocgt','wind' ,'solar','transmission']
                    )

ds_00 = dataset(    './output/archive/prime_euro_00_4D_eta_0.1.csv',
                    data_type='csv',
                    variables=['ocgt','wind' ,'solar','transmission']
                    )

ds_50 = dataset(    './output/archive/prime_euro_50_4D_eta_0.1.csv',
                    data_type='csv',
                    variables=['ocgt','wind' ,'solar','transmission']
                    )
ds_80 = dataset(    './output/archive/prime_euro_80_4D_eta_0.1.csv',
                    data_type='csv',
                    variables=['ocgt','wind' ,'solar','transmission']
                    )
ds_95 = dataset(    './output/archive/prime_euro_95_4D_eta_0.1.csv',
                    data_type='csv',
                    variables=['ocgt','wind' ,'solar','transmission']
                    ) """
#%% new data 


ds_all_01 = dataset([
                    './output/prime_euro_00_storage_5D_eta_0.15.xlsx',
                    './output/prime_euro_50_storage_5D_eta_0.15.xlsx',
                    './output/prime_euro_80_storage_5D_eta_0.15.xlsx',
                    './output/prime_euro_95_storage_5D_eta_0.15.xlsx'],
                    variables=['wind' ,'solar','H2','battery','co2_emission']
                    )

ds_all_05 = dataset([
                    './output/prime_euro_00_storage_5D_eta_0.3.xlsx',
                    './output/prime_euro_50_storage_5D_eta_0.3.xlsx',
                    './output/prime_euro_80_storage_5D_eta_0.3.xlsx',
                    './output/prime_euro_95_storage_5D_eta_0.3.xlsx'],
                    variables=['wind' ,'solar','H2','battery','co2_emission']
                    )

ds_all_10 = dataset([
                    './output/prime_euro_00_storage_5D_eta_0.45.xlsx',
                    './output/prime_euro_50_storage_5D_eta_0.45.xlsx',
                    './output/prime_euro_80_storage_5D_eta_0.45.xlsx',
                    './output/prime_euro_95_storage_5D_eta_0.45.xlsx'],
                    variables=['wind' ,'solar','H2','battery','co2_emission']
                    )

ds_00 = dataset(    './output/prime_euro_00_storage_5D_eta_0.45.xlsx',
                    variables=['wind' ,'solar','H2','battery','co2_emission']
                    )

ds_50 = dataset(    './output/prime_euro_50_storage_5D_eta_0.45.xlsx',
                    variables=['wind' ,'solar','H2','battery','co2_emission']
                    )
ds_80 = dataset(    './output/prime_euro_80_storage_5D_eta_0.45.xlsx',
                    variables=['wind' ,'solar','H2','battery','co2_emission']
                    )
ds_95 = dataset(    './output/prime_euro_95_storage_5D_eta_0.45.xlsx',
                    variables=['wind' ,'solar','H2','battery','co2_emission']
                    )

#%% 

def calc_metrics(ds,max_co2):

    network = pypsaTools.import_network('data/networks/euro_30')

    # CO2 reduction 
    ds.df_secondary_metrics['co2_reduction'] = (1-ds.df_secondary_metrics['co2_emission']/max_co2) *100

    # Land use 
    ds.df_secondary_metrics['land_use'] = calc_land_use(ds,network)

    # Implementation time
    ds.df_secondary_metrics['implementation_time'] = calc_implementation_time(ds,network)

    ds.df_secondary_metrics['gini_capex'] = calc_gini_capex(ds,network)

    #

    return ds

def calc_gini_capex(ds,network):

    #network = pypsaTools.import_network('data/networks/euro_30')
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

    return gini_list


def calc_land_use(ds,network):

    #network = pypsaTools.import_network('data/networks/euro_30')
    buses = network.buses.index
    techs = ['onwind','offwind','solar','ocgt']
    land_use_list = []


    land_fill = dict(onwind= 1/20,  # 20 MW/km2
                solar= 1/145,         # 145 MW/km2
                ocgt=0,
                offwind=0,
                H2=0,
                battery=0)

    for i in range(ds.df_detail.shape[0]):
        part_res = 0 
        for tech in techs:
            part_res += sum(ds.df_detail.iloc[i].filter(like=tech))*land_fill[tech]

        land_use_list.append(part_res)
    return land_use_list



def calc_implementation_time(ds,network):
    #network = pypsaTools.import_network('data/networks/euro_30')
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
        row = ds.df_detail.iloc[i,:]
        expenses = dict.fromkeys(buses,0)

        for bus in buses:
            for tech in techs:
                try:
                    expenses[bus] = expenses[bus] + row[bus+' '+tech]*capex[tech]
                except : 
                    pass
        
        implementation_time = {bus:expenses[bus]/(gdp_dict[bus]*max_use_of_gdp) for bus in buses}

        implementation_time_list.append(max(implementation_time.values()))

    return implementation_time_list

#%%

def plot_gini(x_data,y_data,fig,row,names = ['15% slack', '30% slack', '45% slack',],x_title='Gini coefficient',y_title='CO2 emission reduction [%]'):



    if len(x_data) != len(y_data):
        print('!! Errror !! \n x_data and y_data must be same length')

    colors = ['rgba(0,0,255,0.6)','rgba(0,0,255,0.4)','rgba(0,0,255,0.2)']
    for i in range(len(x_data)):
        # Filter NaN and 0 values 
        x_i = x_data[i][[(a>0)and(b>0) for a,b in zip(x_data[i],y_data[i])]]
        y_i = y_data[i][[(a>0)and(b>0) for a,b in zip(x_data[i],y_data[i])]]

        gini_hull = ConvexHull(np.array([x_i.values,y_i.values]).T)

        x = gini_hull.points[gini_hull.vertices][:,1]
        y = gini_hull.points[gini_hull.vertices][:,0]

        fig.add_trace(go.Scatter(x=np.append(x,x[0]),
                            y=np.append(y,y[0]),
                            mode='lines',
                            showlegend= True if row==1 else False,
                            name=names[i],
                            fillcolor = colors[i],
                            marker_color='rgba(0,0,255,0)',
                            fill='toself'
                            ),row=row,col=1)



    return fig 

#%%

datasets = [ds_all_01,ds_all_05,ds_all_10]
#max_co2 = max(ds_all_01.df_secondary_metrics['co2_emission'])
max_co2 = 1510e6

co2_data = []
gini_data = []
land_use_data = []
gini_capex_data = []
implementation_time_data = [] 
transmission_data = []
wind_data = []
solar_data = []
cost_data = []

for ds in datasets:
    ds = calc_metrics(ds,max_co2)
    ds = calc_interrior(ds)

    co2_data.append(ds.df_interrior['co2_reduction'])
    gini_data.append(ds.df_interrior['gini'])
    land_use_data.append(ds.df_interrior['land_use'])
    gini_capex_data.append(ds.df_interrior['gini_capex'])
    implementation_time_data.append(ds.df_interrior['implementation_time'])
    transmission_data.append(ds.df_interrior['transmission']*1e-3)
    wind_data.append(ds.df_interrior['wind']*1e-3)
    solar_data.append(ds.df_interrior['solar']*1e-3)
    cost_data.append(ds.df_interrior['system_cost'])


co2_optimum = []
gini_optimum = []
gini_capex_optimum = []
land_use_optimum = []
implementation_time_optimum = []
transmission_optimum = []
wind_optimum = []
solar_optimum = []
cost_optimum = []

datasets = [ds_50,ds_80,ds_95]
for ds in datasets:
    ds = calc_metrics(ds,max_co2)
    co2_optimum.append( (1-ds.df_secondary_metrics['co2_emission'][0]/max_co2) *100  )
    gini_optimum.append( ds.df_secondary_metrics['gini'][0] )
    gini_capex_optimum.append( ds.df_secondary_metrics['gini_capex'][0] )
    land_use_optimum.append( ds.df_secondary_metrics['land_use'][0] )
    implementation_time_optimum.append( ds.df_secondary_metrics['implementation_time'][0] )
    transmission_optimum.append( ds.df_sum_var['transmission'][0]*1e-3 )
    wind_optimum.append( ds.df_sum_var['wind'][0]*1e-3 )
    solar_optimum.append( ds.df_sum_var['solar'][0]*1e-3 )
    cost_optimum.append( ds.df_secondary_metrics['system_cost'][0] )

cost_data[1] = cost_data[1].append(cost_data[0],ignore_index=True)
cost_data[2] = cost_data[2].append(cost_data[1],ignore_index=True)

cost_co2 = co2_data.copy()
cost_co2[1] = cost_co2[1].append(cost_co2[0],ignore_index=True)
cost_co2[2] = cost_co2[2].append(cost_co2[1],ignore_index=True)

#%%

fig = make_subplots(rows = 6, cols=1,shared_xaxes=True,shared_yaxes=False,vertical_spacing=0.05,
                    subplot_titles=['<b>a</b>','<b>b</b>','<b>c</b>','<b>d</b>','<b>e</b>','<b>f</b>',]
                    )

fig = plot_gini(gini_data,co2_data,fig,1)
fig = plot_gini(gini_capex_data,co2_data,fig,2)
fig = plot_gini(land_use_data,co2_data,fig,3)
fig = plot_gini(implementation_time_data,co2_data,fig,4)
fig = plot_gini(transmission_data,co2_data,fig,5)
#fig = plot_gini(wind_data,co2_data,fig,6)
fig = plot_gini(cost_data,cost_co2,fig,6)

fig.add_trace(go.Scatter(x=co2_optimum,y=gini_optimum,name='optimum',marker_color='red'),row=1,col=1)
fig.add_trace(go.Scatter(x=co2_optimum,y=gini_capex_optimum,showlegend=False,marker_color='red'),row=2,col=1)
fig.add_trace(go.Scatter(x=co2_optimum,y=land_use_optimum,showlegend=False,marker_color='red'),row=3,col=1)
fig.add_trace(go.Scatter(x=co2_optimum,y=implementation_time_optimum,showlegend=False,marker_color='red'),row=4,col=1)
fig.add_trace(go.Scatter(x=co2_optimum,y=transmission_optimum,showlegend=False,marker_color='red'),row=5,col=1)
#fig.add_trace(go.Scatter(x=co2_optimum,y=wind_optimum,showlegend=False,marker_color='red'),row=6,col=1)
fig.add_trace(go.Scatter(x=co2_optimum,y=cost_optimum,showlegend=False,marker_color='red'),row=6,col=1)



fig.update_yaxes(title_text='Self-sufficiency<br>Gini coefficient',range=[0,0.7],showgrid=False,linecolor='black',title=dict(standoff=20) ,row=1,col=1)
fig.update_yaxes(title_text='Investement<br>Gini coefficient',range=[0,0.7],showgrid=False,row=2,linecolor='black',title=dict(standoff=20),col=1)
fig.update_yaxes(title_text='Land use<br>[km<sup>2</sup>]',range=[0,1e5],showgrid=False,linecolor='black',title=dict(standoff=20),row=3,col=1)
fig.update_yaxes(title_text='Implementation time<br>[years]',range=[0,30],showgrid=False,linecolor='black',title=dict(standoff=20),row=4,col=1)
fig.update_yaxes(title_text='Transmission<br>[GW]',range=[0,2000],showgrid=False,linecolor='black',title=dict(standoff=10),row=5,col=1)
#fig.update_yaxes(title_text='Wind <br> [GW]',range=[0,2000],showgrid=False,linecolor='black',title=dict(standoff=10),row=6,col=1)
fig.update_yaxes(title_text='System cost<br>[Eur]',showgrid=False,linecolor='black',title=dict(standoff=10),row=6,col=1)

fig.update_xaxes(showgrid=False,linecolor='black',row=1,col=1)
fig.update_xaxes(showgrid=False,linecolor='black',row=2,col=1)
fig.update_xaxes(showgrid=False,linecolor='black',row=3,col=1)
fig.update_xaxes(showgrid=False,linecolor='black',row=4,col=1)
fig.update_xaxes(showgrid=False,linecolor='black',row=5,col=1)
fig.update_xaxes(showgrid=False,linecolor='black',row=6,col=1)
fig.update_xaxes(title_text='CO2 reduction [%]',range=[30,100],nticks=8,showgrid=False,linecolor='black',row=6,col=1)

fig.update_layout(width=600,
                height=1000,
                legend=dict(x=0,y=1,bgcolor='rgba(255,255,255,0.8)'),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                #xaxis=dict(showline=True,linecolor='black'),
                #yaxis=dict(showline=True,linecolor='black'),
                )

fig.show()
fig.write_image("figure4.pdf")


#%%


# %%

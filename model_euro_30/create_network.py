#%% make the code as Python 3 compatible as possible
# from __future__ import print_function, division

from vresutils import * 
from vresutils.costdata2 import get_full_cost_CO2
#import vresutils.hydro as vhydro
import vresutils.file_io_helper as io_helper
import vresutils.load as vload 
import pypsa
import datetime
import pandas as pd
import numpy as np
import os,sys
from vresutils import timer
import yaml
import pyomo.environ as pyomo_env
from pyomo.core import ComponentUID
import pickle 
#from vresutils import shapes as vshapes
from math import radians, cos, sin, asin, sqrt

import pyproj
from shapely.ops import transform

import warnings


# %%

discountrate = get_full_cost_CO2.__globals__['discountrate']
USD2013_to_EUR2013 = get_full_cost_CO2.__globals__['USD2013_to_EUR2013']


def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    n = float(n)
    r = float(r)

    if r == 0:
        return 1/n
    else:
        return r/(1. -1./(1.+r)**n)

def get_totalload(timerange=None,nodes=None,interpolate_0=False):
    totload = vload.timeseries_entsoe(directory='data/load/',
                                      years=range(2011,2011+1),
                                      countries=nodes) #MW
    #remove UTC timezone for compatibility with network.snapshots
    totload.index = pd.DatetimeIndex(pd.date_range(str('2011'),str('2012'),freq='H')[:-1])

    if timerange is not None:
        #totload = totload.loc[timerange]
        totload = totload.loc[timerange[0]:timerange[-1]]
    if nodes is not None:
        totload = totload[nodes]
    if interpolate_0:
        # remove occasional hours with zero load by linear interpolation
        totload = totload.where(totload != 0,np.nan).interpolate(limit=3)
        assert not totload.isnull().any().any(), 'Trying to interpolate more than 3 consecutive missing data.'
    return totload


def haversine(p1,p2):
    """Calculate the great circle distance in km between two points on
    the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Function used to generate country centroid coordinates in xy
def country_centroids(node):
        data = pd.read_csv('data/country_centroids.csv',sep=',',encoding='latin-1')

        lon = data[data.country==node].longitude
        lat = data[data.country==node].latitude
        return lon, lat

def solveit(options):

    with timer('initialize model'):
        network=init_model(options)
    with timer('solving model'):
        solve_model(network)
    if options['save_res']:
        with timer('saving results'):
            save_results(network)

    return network

#%%


dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
os.chdir(dir_path)
options_file_name = "options_test.yml"
print(options_file_name)
options = yaml.load(open(dir_path+options_file_name,"r"),Loader=yaml.FullLoader)
options['co2_reduction'] = 0.5

#Build the Network object, which stores all other objects
network = pypsa.Network()

#network.opf_keep_files=False
#network.options=options

#load graph
nodes = pd.read_csv("data/graph/nodes.csv",header=None,squeeze=True)
edges = pd.read_csv("data/graph/edges.csv",header=None)

#set times
network.set_snapshots(pd.date_range(str(options['tmin']),str(options['tmax']),freq='H',closed='left'))

represented_hours = network.snapshot_weightings.sum()
Nyears= represented_hours/8760.


#load renewables time series
variable_generator_kinds = {'onwind':'onwind','offwind':'offwind','solar':'solar'}
if options['split_onwind']:
    variable_generator_kinds.update({'onwind':'onwind_split'})

p_max_pu_folder='data/renewables/store_p_max_pu_betas/'

if options['split_onwind']:
    split_countries = pd.Series.from_csv(os.path.join(p_max_pu_folder,'onwind_split_countries.csv'))

def get_p_max_pu(path,timerange):
    pmpu = pd.read_pickle(path)
    pmpu.index = pd.DatetimeIndex(pmpu.index.values)
    pmpu = pmpu.loc[timerange]
    return pmpu

p_max_pu = {kind: get_p_max_pu(os.path.join(p_max_pu_folder,'p_max_pu_{kname}_beta{beta}.pickle'.format(kname=kname,beta=options['beta_layout'])),network.snapshots)
        for kind,kname in variable_generator_kinds.items()}

p_nom_max_folder = 'data/renewables/store_p_nom_max/'

p_nom_max = {kind: pd.read_pickle(os.path.join(p_nom_max_folder,'p_nom_max_{kname}_beta{beta}.pickle'.format(kname=kname,beta=options['beta_layout'])))
        for kind,kname in variable_generator_kinds.items()}


#define costs for renewables and conventionals (including CO2 costs)
cost_df = get_full_cost_CO2(options['costref'],options['CO2price'],
        filename='data/costs/costdata_new.xls') #in [Eur/MW] [Eur/MWh]

#%% add carriers
conventionals = ['ocgt']
for ftyp in conventionals:
    network.add("Carrier",ftyp,co2_emissions=cost_df.at[ftyp,'CO2intensity']) # in t_CO2/MWht
network.add("Carrier","onwind")
network.add("Carrier","offwind")
network.add("Carrier","solar")

if options['add_H2_storage']:
    network.add("Carrier","H2")
if options['add_battery_storage']:
    network.add("Carrier","battery")

# %%

totalload_df = get_totalload(network.snapshots,nodes,interpolate_0=True) #MW

# %%


for node in nodes.values:
    # Add node
    x,y = country_centroids(node)
    network.add("Bus",node,x=x.values[0],y=y.values[0])

    # add load 
    network.add("Load",node,
            bus=node,
            p_set=totalload_df[node]
            )

    #add renewables

    #add onshore wind
    network.add("Generator",
        node+" onwind",
        p_nom_extendable=True,
        bus=node,
        carrier="onwind",
        type='wind',
        p_nom_max=p_nom_max['onwind'][node],
        capital_cost = Nyears*cost_df.at['onwind','capital'],
        p_max_pu=p_max_pu['onwind'][node],
        marginal_cost=cost_df.at['onwind','marginal'],
        )

    #add offshore wind
    if p_nom_max['offwind'][node]>0:
        network.add("Generator",
            node+" offwind",
            p_nom_extendable=True,
            bus=node,
            carrier="offwind",
            type='wind',
            p_nom_max=p_nom_max['offwind'][node],
            capital_cost = Nyears*cost_df.at['offwind','capital'],
            p_max_pu=p_max_pu['offwind'][node],
            marginal_cost=cost_df.at['offwind','marginal'],
            )

    #add solar
    network.add("Generator",
            node+" solar",
            p_nom_extendable=True,
            bus=node,
            carrier="solar",
            type='solar',
            p_nom_max=p_nom_max['solar'][node],
            capital_cost = Nyears*cost_df.at['solar','capital'],
            p_max_pu=p_max_pu['solar'][node] * options['solar_cf_correction'],
            marginal_cost=cost_df.at['solar','marginal'],
            )

    #add conventionals
    for ftyp in conventionals:
        # add generators
        network.add("Generator",
                node+" "+ftyp,
                p_nom_extendable=True,
                bus=node,
                carrier=ftyp,
                type='ocgt',
                #dispatch="flexible",
                capital_cost=Nyears*cost_df.at[ftyp,'capital'],
                marginal_cost=cost_df.at[ftyp,'marginal'],
                efficiency=cost_df.at[ftyp,'efficiency'],
                )


    #add storage
    if options['add_H2_storage']:
        #add H2 storage
        H2_max_hours = network.options['H2_max_hours']
        network.add("StorageUnit",
                node+" H2",
                bus=node,
                carrier="H2",
                p_nom_extendable=True,
                #e_nom_extendable=True,
                cyclic_state_of_charge=True,
                #max_hours_fixed=True,#False,
                max_hours=H2_max_hours,
                p_max_pu=1,
                p_min_pu=-1,
                efficiency_store=0.75,
                efficiency_dispatch=0.58,
                capital_cost=((annuity(20.,discountrate)*737.+12.2)*1000.*USD2013_to_EUR2013*Nyears
                    + H2_max_hours * (annuity(20.,discountrate)*11.2*1000.*USD2013_to_EUR2013*Nyears)),  # Eur/MW/year*years
                marginal_cost=options['marginal_cost_storage']
                )

    if options['add_battery_storage']:
        battery_max_hours = network.options['battery_max_hours']
        #add battery storage (Lithium titanate)
        network.add("StorageUnit",
                node+" battery",
                bus=node,
                carrier="battery",
                p_nom_extendable=True,
                cyclic_state_of_charge=True,
                max_hours=battery_max_hours,
                efficiency_store=0.9,
                efficiency_dispatch=0.9,
                capital_cost=((annuity(20.,discountrate)*411.+12.3)*1000.*USD2013_to_EUR2013*Nyears
                    + battery_max_hours * (annuity(20.,discountrate)*192.*1000.*USD2013_to_EUR2013*Nyears)), # Eur/MW/year*years
                marginal_cost=options['marginal_cost_storage']
                )
#%% Add lines 
for name0,name1 in edges.values:
    
    name = "{}-{}".format(name0,name1)
    length = haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                    [network.buses.at[name1,"x"],network.buses.at[name1,"y"]])
    capital_cost = ((options['line_cost_factor']*length*400*1.25+150000.) \
                * 1.5 *1.02 \
                *Nyears*annuity(40.,discountrate))

    network.add("Link", name, bus0=name0, bus1=name1,
                p_nom_extendable=True,
                p_nom=0,
                #p_min_pu=-1,
                length=length,
                capital_cost=capital_cost)

# %%

if options['co2_reduction'] is not 0:
    #network.co2_limit = options['co2_reduction']*1.55e9*Nyears
    #unbound_emission = 1151991057.2540295
    #unbound_emission  =  571067122.7405636
    unbound_emission  =  1510e6
    target = (1-options['co2_reduction'])*unbound_emission
    network.add("GlobalConstraint","co2_limit",
            sense="<=",
            carrier_attribute="co2_emissions",
            constant=target)

# %%
network.lopf(solver_name='gurobi')
old_objective_value = network.objective
model = network.model
#%%
MGA_slack = 0.1
# Add the MGA slack constraint.
model.mga_constraint = pyomo_env.Constraint(expr=model.objective.expr <= 
                                        (1 + MGA_slack) * old_objective_value)

# Saving model as .lp file
_, smap_id = model.write("test.lp",)
# Creating symbol map, such that variables can be maped back from .lp file to pyomo model
symbol_map = model.solutions.symbol_map[smap_id]
#%%
tmp_buffer = {} # this makes the process faster
symbol_cuid_pairs = dict(
        (symbol, ComponentUID(var_weakref(), cuid_buffer=tmp_buffer))
        for symbol, var_weakref in symbol_map.bySymbol.items())


# %% Pickeling variable pairs 
with open('var_pairs.pickle', 'wb') as handle:
    pickle.dump(symbol_cuid_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%

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

#from vresutils import shapes as vshapes
from math import radians, cos, sin, asin, sqrt

import pyproj
from shapely.ops import transform

import warnings


#import cPickle as pickle
#%%
#country_shapes = vshapes.countries()

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

def init_model(options):


    #Build the Network object, which stores all other objects
    network = pypsa.Network()

    network.opf_keep_files=False
    network.options=options

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

    #add carriers
    conventionals = ['ocgt']
    for ftyp in conventionals:
        network.add("Carrier",ftyp,co2_emissions=cost_df.at[ftyp,'CO2intensity']) # in t_CO2/MWht
    network.add("Carrier","onwind")
    network.add("Carrier","offwind")
    network.add("Carrier","solar")
    if options['add_PHS']:
        network.add("Carrier","PHS")
    if options['add_hydro']:
        network.add("Carrier","hydro")
    if options['add_ror']:
        network.add("Carrier","ror")
    if options['add_H2_storage']:
        network.add("Carrier","H2")
    if options['add_battery_storage']:
        network.add("Carrier","battery")
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
    #else :
        # Constraint is in order to access emission data
    #    target = np.inf
    #    network.add("GlobalConstraint","co2_limit",
    #          sense="<=",
    #          carrier_attribute="co2_emissions",
    #          constant=target)

    #load hydro data
    if options['add_PHS'] or options['add_hydro']:
        hydrocapa_df = vhydro.get_hydro_capas(fn='data/hydro/emil_hydro_capas.csv')
        if options['add_ror']:
            ror_share = vhydro.get_ror_shares(fn='data/hydro/ror_ENTSOe_Restore2050.csv')
        else:
            ror_share = pd.Series(0,index=hydrocapa_df.index)

    if options['add_hydro']:
        inflow_df = vhydro.get_hydro_inflow(inflow_dir='data/hydro/inflow/')*1e3 # GWh/h to MWh/h
        # if Hydro_Inflow from Restore2050 is not available, use alternative dataset:
        #inflow_df = vhydro.get_inflow_NCEP_EIA().to_series().unstack(0) #MWh/h

        # select only nodes that are in the network
        #inflow_df = inflow_df.loc[network.snapshots,nodes].dropna(axis=1)
        inflow_df = inflow_df.loc[network.snapshots].dropna(axis=1)


    #load demand data
    totalload_df = get_totalload(network.snapshots,nodes,interpolate_0=True) #MW

    # Function used to generate country centroid coordinates in xy
    def country_centroids(node):
        data = pd.read_csv('data/country_centroids.csv',sep=',',encoding='latin-1')

        lon = data[data.country==node].longitude
        lat = data[data.country==node].latitude
        return lon, lat


    for node in nodes.values:
        network.add("Bus",node)

        # add load timeseries
        network.add("Load",node,
                bus=node,
                p_set=totalload_df[node]
                )

        x,y = country_centroids(node)

        # lat/lon of node
        network.buses.at[node,"x"] = x.values[0]
        network.buses.at[node,"y"] = y.values[0]


        #add renewables

        #add onshore wind
        if options['split_onwind'] and (node in split_countries):
            charr = p_max_pu['onwind'].keys()
            splitnodes = charr[charr.str.startswith(node)] #select all nodes with names that contain `node`, e.g., `DE` -> `DE0`,`DE1`,`DE2`
        else:
            splitnodes = [node]

        for this_node in splitnodes:
            network.add("Generator",
                this_node+" onwind",
                p_nom_extendable=True,
                bus=node,
                carrier="onwind",
                type='wind',
                p_nom_max=p_nom_max['onwind'][this_node],
                capital_cost = Nyears*cost_df.at['onwind','capital'],
                p_max_pu=p_max_pu['onwind'][this_node],
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

        if options['add_PHS']:
            # pure pumped hydro storage, fixed, 6h energy by default, no inflow

            hcapa = hydrocapa_df.loc[node]
            if hcapa['p_nom_store[GW]'] > 0:
                if options['hydro_capital_cost']:
                    cc=Nyears*cost_df.at['PHS','capital']
                else: cc=0.
                network.add("StorageUnit",
                    node+" PHS",
                    bus=node,
                    carrier="PHS",
                    p_nom_extendable=False,
                    p_nom=hcapa['p_nom_store[GW]']*1000., #from GW to MW
                    max_hours=options['PHS_max_hours'],
                    efficiency_store=np.sqrt(cost_df.at['PHS','efficiency']),
                    efficiency_dispatch=np.sqrt(cost_df.at['PHS','efficiency']),
                    cyclic_state_of_charge=True,
                    capital_cost = cc,
                    marginal_cost=options['marginal_cost_storage']
                    )



        if options['add_hydro']:
            # inflow hydro:
            #  * run-of-river if E_s=0
            #  * reservoir
            #  * could include mixed pumped, if 0>p_min_pu_fixed=p_pump*p_nom

            hcapa = hydrocapa_df.loc[node]

            pnom = (1.-ror_share[node])* hcapa['p_nom_discharge[GW]']*1000. #GW to MW
            if pnom > 0:

                if options['hydro_max_hours'] is None:
                    max_hours=hcapa['E_store[TWh]']*1e6/pnom #TWh to MWh
                else:
                    max_hours=options['hydro_max_hours']


                try:
                    inflow = (1.-ror_share[node])*inflow_df[node]
                except KeyError:
                    inflow = 0

                if options['hydro_capital_cost']:
                    cc=Nyears*cost_df.at['hydro','capital']
                else: cc=0.

                network.add("StorageUnit",
                    node+" hydro",
                    bus=node,
                    carrier="hydro",
                    p_nom_extendable=False,
                    p_nom=pnom,
                    max_hours=max_hours,
                    p_max_pu=1,  #dispatch
                    p_min_pu=0.,  #store
                    efficiency_dispatch=1,
                    efficiency_store=0,
                    inflow=inflow,
                    cyclic_state_of_charge=True,
                    capital_cost = cc,
                    marginal_cost=options['marginal_cost_storage']
                    )

        if options['add_ror']:
            if (ror_share[node] > 0) and (node in inflow_df.keys()):

                hcapa = hydrocapa_df.loc[node]
                pnom = ror_share[node]* hcapa['p_nom_discharge[GW]']*1000. #GW to MW
                inflow_pu = ror_share[node]*inflow_df[node] / pnom
                inflow_pu[inflow_pu>1]=1. #limit inflow per unit to one, i.e, excess inflow is spilled here

                if options['hydro_capital_cost']:
                    cc=Nyears*cost_df.at['ror','capital']
                else: cc=0.

                network.add("Generator",
                    node+" ror",
                    bus=node,
                    carrier="ror",
                    p_nom_extendable=False,
                    p_nom=pnom,
                    p_max_pu=inflow_pu,
                    capital_cost = cc,
                    marginal_cost=options['marginal_cost_storage']
                    )



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


    #add lines

    for name0,name1 in edges.values:
        if network.options['no_lines']: break
        name = "{}-{}".format(name0,name1)
        network.add("Link", name, bus0=name0, bus1=name1,
                    p_nom_extendable=True,
                    p_min_pu=-1,
                    length=haversine([network.buses.at[name0,"x"],network.buses.at[name0,"y"]],
                                     [network.buses.at[name1,"x"],network.buses.at[name1,"y"]]))

        if options['line_volume_limit_max'] is not None:
            network.links.at[name,"capital_cost"] = Nyears*0.01 # Set line costs to ~zero because we already restrict the line volume
        else:
            network.links.at[name,"capital_cost"] = ((options['line_cost_factor']*network.links.at[name,"length"]*400*1.25+150000.) \
                    * 1.5 *1.02 \
                    *Nyears*annuity(40.,discountrate))
            # 1.25 because lines are not straigt, 150000 is per MW cost of
            # converter pair for DC line,
            # n-1 security is approximated by an overcapacity factor 1.5
            # FOM of 2%

    return network

#%%
def extra_functionality(network,snapshots):
    #force solver to also give us the reduced prices (duals of variable range constraints)
    network.model.rc = pypsa.opf.Suffix(direction=pypsa.opf.Suffix.IMPORT)

    # add a very small penalty to (one hour of) the state of charge of
    # non-extendable storage units -- this ensures that the storage is empty in
    # (at least) one hour
    if not hasattr(network, 'epsilon'):
        Nyears= network.snapshot_weightings.sum()/8760.
        network.epsilon = 1e-2 * Nyears

    fix_sus = network.storage_units[~ network.storage_units.p_nom_extendable]
    network.model.objective.expr += sum(network.epsilon*network.model.state_of_charge[su,network.snapshots[0]] for su in fix_sus.index)

    if network.options['line_volume_limit_max'] is not None:
        network.model.line_volume_limit = pypsa.opt.Constraint(expr=sum(network.model.link_p_nom[link]*network.links.at[link,"length"] for link in network.links.index if link[2:3] == "-") <= network.options['line_volume_limit_factor']*network.options['line_volume_limit_max'])
#%%
def solve_model(network):
    solver_name = network.options['solver_name']
    solver_io = network.options['solver_io']
    solver_options = network.options['solver_options']
    check_logfile_option(solver_name,solver_options)
    with timer('lopf optimization'):
        network.lopf(network.snapshots[0],solver_name=solver_name,solver_io=solver_io,solver_options=solver_options,extra_functionality=extra_functionality,keep_files=network.opf_keep_files,formulation=network.options['formulation'])

    # save the shadow prices of some constraints
    network.shadow_prices = {}
    for constr_name in ['line_volume_limit','co2_constraint']:
        try:
            network.shadow_prices.update({constr_name :
                network.model.dual[getattr(network.model, constr_name)]})
        except AttributeError:
            print('NB: network does not have a constraint named `{}`'.format(constr_name))
    return network


def save_results(network):

    results_dir_name = os.path.join(network.options['results_dir_name'],network.options['run_name'])
    results_suffix = network.options['results_suffix']

    if results_dir_name != '':
        io_helper.ensure_mkdir(results_dir_name)


    techstr = 'wWsg'+''.join(np.array(['r','p','H','b'])[np.array([network.options['add_hydro'],network.options['add_PHS'],network.options['add_H2_storage'],network.options['add_battery_storage']])])


    results_name = '{}-CO{}-T{}_{}-{}'.format(
        network.options['costref'],
        network.options['CO2price'],
        network.options['tmin'],
        network.options['tmax'],
        techstr,
        )

    results_folder_name = os.path.join(results_dir_name,
            '-'.join(filter(None,[
                results_name,
                results_suffix,
                ]))
            )

    io_helper.ensure_mkdir(results_folder_name)



    #move logfile and scripts into the folder
    solver_file = os.path.join(results_dir_name,'solver_{suffix}.log'.format(suffix=results_suffix))
    opt_ws_file = os.path.join(results_dir_name,'opt_ws_network_{suffix}.py'.format(suffix=results_suffix))
    options_file = os.path.join(results_dir_name,'options_{suffix}.yml'.format(suffix=results_suffix))
    parba_file = os.path.join(results_dir_name,'parameter_batch_{suffix}.py'.format(suffix=results_suffix))

    if os.path.isfile(solver_file):
       os.system('mv {} {}'.format(solver_file,os.path.join(results_folder_name,os.path.basename(solver_file))))

    if os.path.isfile(parba_file):
       os.system('mv {} {}'.format(parba_file,os.path.join(results_folder_name,os.path.basename(parba_file))))

    if os.path.isfile(options_file):
       os.system('mv {} {}'.format(options_file,os.path.join(results_folder_name,os.path.basename(options_file))))

    if os.path.isfile(opt_ws_file):
       os.system('mv {} {}'.format(opt_ws_file,os.path.join(results_folder_name,os.path.basename(opt_ws_file))))


    network.export_to_csv_folder(results_folder_name)

    dicts = ['options','shadow_prices']
    for dic in dicts:
        export_dict_to_csv(getattr(network, dic),os.path.join(results_folder_name,dic+'.csv'))



def export_dict_to_csv(dic,filename,mode='w'):
    for k, v in dic.items():
        if v is None:
            dic[k] = 'None'
    import csv
    with open(filename, mode=mode) as outfile:
        writer = csv.DictWriter(outfile,dic.keys())
        writer.writeheader()
        writer.writerow(dic)
def import_dict_from_csv(filename):
    '''Somehow this takes care of unit conversion'''
    df=pd.read_csv(filename)
    dic = df.where((pd.notnull(df)),None).T[0].to_dict()
    return dic

def check_logfile_option(solver_name,solver_options):
    #make sure to use right keyword for each solver
    #'logfile' for gurobi
    #'log' for glpk
    if 'logfile' in solver_options and solver_name == 'glpk':
        solver_options['log'] = solver_options.pop('logfile')
    elif 'log' in solver_options and solver_name == 'gurobi':
        solver_options['logfile'] = solver_options.pop('log')


#%%

if __name__ == '__main__':
    
    dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
    os.chdir(dir_path)
    options_file_name = "options_MGA_storage.yml"
    print(options_file_name)
    options = yaml.load(open(dir_path+options_file_name,"r"),Loader=yaml.FullLoader)

    co2_reductions = [0,0.5,0.8,0.95]
    for co2_red in co2_reductions :

        options['co2_reduction'] = co2_red
    
        network = init_model(options)
        network = solve_model(network)
     
        network.export_to_hdf5('./network_csv/euro_'+'{:02.0f}_storage'.format(co2_red*100))



# %% Testing of the code 
    if False : 
        dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
        os.chdir(dir_path)
        options_file_name = "options_test.yml"
        print(options_file_name)
        options = yaml.load(open(dir_path+options_file_name,"r"),Loader=yaml.FullLoader)
        options['co2_reduction'] = 0.5
        co2_red = 0.5
        network = init_model(options)

        network = solve_model(network)

        network.export_to_hdf5('./network_csv/euro_test'.format(co2_red*100))
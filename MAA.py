#%%
# Author: Tim Pedersen
# Contact: timtoernes@gmail.com

import os 
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import warnings
warnings.simplefilter("ignore")
import logging
logging.basicConfig(level=logging.ERROR)
import pypsa
#import gurobipy
import time
import numpy as np
import sys
import yaml
import pyomo.environ as pyomo_env
import pandas as pd
from scipy.spatial import ConvexHull
import multiprocessing as mp
from multiprocessing import Lock, Process, Queue, current_process
import queue # imported for using queue.Empty exception
sys.path.append(os.getcwd())
import gc
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective
#import pypsa_tools as pt


#%% Solutions class 
class solutions:
    # the solutions class contains all nececary data for all MGA solutions
    # The class also contains functions to append new solutions and to save the results

    def __init__(self,network):
        self.old_objective = network.objective
        self.sum_vars = self.calc_sum_vars(network)
        self.gen_p =    pd.DataFrame(data=[network.generators.p_nom_opt],index=[0])
        self.gen_E =    pd.DataFrame(data=[network.generators_t.p.sum()],index=[0])
        self.store_p =  pd.DataFrame(data=[network.storage_units.p_nom_opt],index=[0])
        self.store_E =  pd.DataFrame(data=[network.storage_units_t.p.sum()],index=[0])
        self.links =    pd.DataFrame(data=[network.links.p_nom_opt],index=[0])
        self.secondary_metrics = self.calc_secondary_metrics(network)
        self.objective = pd.DataFrame()

        self.df_list = {'gen_p':self.gen_p,
                        'gen_E':self.gen_E,
                        'store_E':self.store_E,
                        'store_p':self.store_p,
                        'links':self.links,
                        'sum_vars':self.sum_vars,
                        'secondary_metrics':self.secondary_metrics}

        try :
            co2_emission = [constraint.body() for constraint in network.model.global_constraints.values()][0]
        except :
            co2_emission = 0 
        

    def append(self,network):
        # Append new data to all dataframes
        self.sum_vars = self.sum_vars.append(self.calc_sum_vars(network),ignore_index=True)
        self.gen_p =    self.gen_p.append([network.generators.p_nom_opt],ignore_index=True)
        self.links =    self.gen_p.append([network.links.p_nom_opt],ignore_index=True)
        self.gen_E =    self.gen_E.append([network.generators_t.p.sum()],ignore_index=True)
        self.secondary_metrics = self.secondary_metrics.append(self.calc_secondary_metrics(network),ignore_index=True)

    def calc_secondary_metrics(self,network):
        # Calculate secondary metrics
        gini = self.calc_gini(network)
        co2_emission = self.calc_co2_emission(network)
        system_cost = self.calc_system_cost(network)
        autoarky = self.calc_autoarky(network)
        return pd.DataFrame({'system_cost':system_cost,'co2_emission':co2_emission,'gini':gini,'autoarky':autoarky},index=[0])

    def calc_sum_vars(self,network):
        sum_data = dict(network.generators.p_nom_opt.groupby(network.generators.type).sum())
        sum_data['transmission'] = network.links.p_nom_opt.sum()
        sum_data['co2_emission'] = self.calc_co2_emission(network)
        sum_data.update(network.storage_units.p_nom_opt.groupby(network.storage_units.carrier).sum())
        sum_vars = pd.DataFrame(sum_data,index=[0])
        return sum_vars

    def put(self,network):
    # add new data to the solutions queue. This is used when new data is added from 
    # sub-process, when using multiprocessing 
        try :
            self.queue.qsize()
        except : 
            print('creating queue object')
            self.queue = Queue()

        part_result = solutions(network)
        self.queue.put(part_result,block=True,timeout=120)

    def init_queue(self):
        # Initialize results queue 
        try :
            self.queue.qsize()
        except : 
            self.queue = Queue()

    def merge(self):
        # Merge all solutions put into the solutions queue into the solutions dataframes
        merge_num = self.queue.qsize()
        while not self.queue.empty() :
            part_res = self.queue.get(60)
            self.gen_E = self.gen_E.append(part_res.gen_E,ignore_index=True)
            self.gen_p = self.gen_p.append(part_res.gen_p,ignore_index=True)
            self.store_E = self.store_E.append(part_res.store_E,ignore_index=True)
            self.store_p = self.store_p.append(part_res.store_p,ignore_index=True)
            self.links = self.links.append(part_res.links,ignore_index=True)
            self.sum_vars = self.sum_vars.append(part_res.sum_vars,ignore_index=True)
            self.secondary_metrics = self.secondary_metrics.append(part_res.secondary_metrics,ignore_index=True)
        print('merged {} solution'.format(merge_num))

    def save_xlsx(self,file='save.xlsx'):
        # Store all dataframes als excel file
        self.df_list = {'gen_p':self.gen_p,
                'gen_E':self.gen_E,
                'store_E':self.store_E,
                'store_p':self.store_p,
                'links':self.links,
                'sum_vars':self.sum_vars,
                'secondary_metrics':self.secondary_metrics}

        writer = pd.ExcelWriter(file)
        sheet_names =  ['gen_p','gen_E','links','sum_var','secondary_metrics']
        for i, df in enumerate(self.df_list):
            self.df_list[df].to_excel(writer,df)
        writer.save() 
        print('saved {}'.format(file))

    def calc_gini(self,network):
    # This function calculates the gini coefficient of a given PyPSA network. 
        bus_total_prod = network.generators_t.p.sum().groupby(network.generators.bus).sum()
        load_total= network.loads_t.p_set.sum()

        rel_demand = load_total/sum(load_total)
        rel_generation = bus_total_prod/sum(bus_total_prod)
        
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
        return gini

    def calc_autoarky(self,network):
        # calculates the autoarky of a model solution 
        # autoarky is calculated as the mean self sufficiency (energy produced/energy consumed) of all countries in all hours
        mean_autoarky = []
        for snap in network.snapshots:
            hourly_load = network.loads_t.p_set.loc[snap]
            hourly_autoarky = network.generators_t.p.loc[snap].groupby(network.generators.bus).sum()/hourly_load
            hourly_autoarky_corected = hourly_autoarky.where(hourly_autoarky<1,1)
            mean_autoarky.append(np.mean(hourly_autoarky_corected))
        return np.mean(mean_autoarky)

    def calc_co2_emission(self,network):
            #CO2
        id_ocgt = network.generators.index[network.generators.type == 'ocgt']
        co2_emission = network.generators_t.p[id_ocgt].sum().sum()*network.carriers.co2_emissions['ocgt']/network.generators.efficiency.loc['AT ocgt']
        co2_emission
        return co2_emission

    def calc_system_cost(self,network):
        #Cost
        capital_cost = sum(network.generators.p_nom_opt*network.generators.capital_cost) + sum(network.links.p_nom_opt*network.links.capital_cost) + sum(network.storage_units.p_nom_opt * network.storage_units.capital_cost)
        marginal_cost = network.generators_t.p.groupby(network.generators.type,axis=1).sum().sum() * network.generators.marginal_cost.groupby(network.generators.type).mean()
        total_system_cost = marginal_cost.sum() + capital_cost
        return total_system_cost

#%% Helper functions

def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'::
    unit_vector = lambda vector: vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

#%%

def inital_solution(network,options):
    # This function performs the initial optimization of the techno-economic PyPSA model
    print('starting initial solution')
    timer = time.time()
    logging.disable()
    # Solving network
    network.lopf(network.snapshots, 
                solver_name='gurobi',
                solver_options={'LogToConsole':0,
                                            'crossover':0,
                                            #'presolve': 2,
                                            #'NumericFocus' : 3,
                                            'method':2,
                                            'threads':options['cpus'],
                                            #'NumericFocus' : numeric_focus,
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                pyomo=False,
                keep_references=True,
                formulation='kirchhoff',
                solver_dir = options['tmp_dir']
                ),
    # initializing solutions class, to keep all network data
    sol = solutions(network)
    print('finished initial solution in {} sec'.format(time.time()-timer))
    return network,sol

#%% MGA function 

def mga_constraint(network,snapshots,options):
    scale = 1e-6
    # This function creates the MGA constraint 
    gen_capital_cost   = linexpr((scale*network.generators.capital_cost,get_var(network, 'Generator', 'p_nom'))).sum()
    gen_marginal_cost  = linexpr((scale*network.generators.marginal_cost,get_var(network, 'Generator', 'p'))).sum().sum()
    store_capital_cost = linexpr((scale*network.storage_units.capital_cost,get_var(network, 'StorageUnit', 'p_nom'))).sum()
    link_capital_cost  = linexpr((scale*network.links.capital_cost,get_var(network, 'Link', 'p_nom'))).sum()
    # total system cost
    cost_scaled = join_exprs(np.array([gen_capital_cost,gen_marginal_cost,store_capital_cost,link_capital_cost]))
    # MGA slack
    if options['mga_slack_type'] == 'percent':
        slack = network.old_objective*options['mga_slack']+network.old_objective
    elif options['mga_slack_type'] == 'fixed':
        slack = options['baseline_cost']*options['mga_slack']+options['baseline_cost']

    define_constraints(network,cost_scaled,'<=',slack*scale,'GlobalConstraint','MGA_constraint')


def mga_objective(network,snapshots,direction,options):
    mga_variables = options['mga_variables']
    expr_list = []
    for i,variable in enumerate(mga_variables):
        if variable == 'transmission':
            expr_list.append(linexpr((direction[i],get_var(network,'Link','p_nom'))).sum())
        if variable == 'co2_emission':
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p').filter(network.generators.index[network.generators.type == 'ocgt']))).sum().sum())
        elif variable == 'H2' or variable == 'battery':
            expr_list.append(linexpr((direction[i],get_var(network,'StorageUnit','p_nom').filter(network.storage_units.index[network.storage_units.carrier == variable]))).sum())
        else : 
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p_nom').filter(network.generators.index[network.generators.type == variable]))).sum())

    mga_obj = join_exprs(np.array(expr_list))
    write_objective(network,mga_obj)

def extra_functionality(network,snapshots,direction,options):
    mga_constraint(network,snapshots,options)
    mga_objective(network,snapshots,direction,options)


def solve(network,options,direction):
    stat = network.lopf(network.snapshots,
                            pyomo=False,
                            solver_name='gurobi',
                            solver_options={'LogToConsole':0,
                                            'crossover':0,
                                            #'presolve': 0,
                                            'ObjScale' : 1e6,
                                            'NumericFocus' : 3,
                                            'method':2,
                                            'threads':int(np.ceil(options['cpus']/options['number_of_processes'])),
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                            keep_references=True,
                            skip_objective=True,
                            formulation='kirchhoff',
                            solver_dir = options['tmp_dir'],
                            extra_functionality=lambda network,snapshots: extra_functionality(network,snapshots,direction,options))
    return network,stat

def job(tasks_to_accomplish,sol,finished_processes,options):
    # This function starts a job in a parallel thred. 
    # Jobs are pulled from the job queue and results are
    # returned in the results queue
    proc_name = current_process().name
    network = import_network(options,tmp_network=True)
    while True:
        try:
            #try to get task from the queue. get_nowait() function will 
            #raise queue.Empty exception if the queue is empty. 
            #queue(False) function would do the same task also.
            direction = tasks_to_accomplish.get(False)
            direction = direction*1e2
        except queue.Empty:
            print('no more jobs')
            break
        else:
            logging.disable()
            network.old_objective = sol.old_objective
            try : 
                max_trys = 4
                for i in range(max_trys):
                    network.old_objective = sol.old_objective
                    network,stat = solve(network,options,direction)
                    print(stat)
                    if stat[1] == 'numeric':
                        print(direction)
                        direction = direction * 1e2
                        print('trying {}nd time'.format(i+2))
                    else :
                        sol.put(network)
                        break

                
            except Exception as e:
                print('did not solve {} direction, process {}'.format(direction,proc_name))
                print(e)
    print('finishing process {}'.format(proc_name))
    finished_processes.put('done')
    return

def start_parallel_pool(directions,network,options,sol):
    # This function will start a pool of jobs using all available cores on the machine
    # Each job is assigned two cores
    number_of_processes = int(os.cpu_count()/2 if len(directions)>os.cpu_count()/2 else len(directions))
    options['number_of_processes'] = number_of_processes
    print('starting {} processes for {} jobs'.format(number_of_processes,len(directions)))
    tasks_to_accomplish = Queue()
    finished_processes = Queue()
    sol.init_queue()
    processes = []
    network.export_to_csv_folder(options['tmp_dir']+'network/')

    # Adding tasks to task queue
    for direction in directions:
        tasks_to_accomplish.put(direction)
    time.sleep(1) # Wait for queue to finsih filling 

    # creating processes
    for w in range(number_of_processes):
        if not tasks_to_accomplish.empty():
            p = Process(target=job, args=(tasks_to_accomplish,sol,finished_processes,options))
            processes.append(p)
            p.start()
            print('{} started'.format(p.name))
        else :
            print('no more jobs - not starting any more processes')

    # wait for all processes to finish
    print('waiting for processes to finish ')
    wait_timer = time.time()
    wait_timeout = 360000
    while not len(processes) == finished_processes.qsize():
        if time.time()-wait_timer > wait_timeout :
            print('wait timed out')
            break
        time.sleep(5)

    # Join all sub proceses
    for p in processes:
        print('waiting to join {}'.format(p.name))
        try :
            p.join(1)
        except :
            p.terminate()
            p.join(60)
            print('killed {}'.format(p.name))
        else :
            print('joined {}'.format(p.name))

    # Merge results from subproces
    sol.merge()

    # Kill any zombie proceses
    for p in processes:
        p.kill()
        time.sleep(1)
        #p.close()
    
    # Close all queues 
    tasks_to_accomplish.close()
    tasks_to_accomplish.join_thread()
    finished_processes.close()
    finished_processes.join_thread()
    gc.collect()
    return sol

def filter_directions(directions,directions_searched):
    # Filter already searched directions out if the angle between the new vector and any 
    # previously sarched vector is less than 1e-2 radians
    obsolete_directions = []
    for direction,i in zip(directions,range(len(directions))):
        if any([abs(angle_between(dir_searched,direction))<1e-2  for dir_searched in directions_searched]) :
            obsolete_directions.append(i)
    directions = np.delete(directions,obsolete_directions,axis=0)

    if len(directions)>1000:
        directions = directions[0:1000]
    return directions

def run_mga(network,sol,options):
    # This is the real MGA function
    MGA_convergence_tol = options['mga_convergence_tol']
    dim=len(options['mga_variables'])
    old_volume = 0 
    epsilon_log = [1]
    directions_searched = np.empty([0,dim])
    hull = None
    computations = 0

    while not all(np.array(epsilon_log[-2:])<MGA_convergence_tol) : # The last two itterations must satisfy convergence tollerence
        # Generate list of directions to search in for this batch
        if options['random_directions'] == True:
            n_rand_dirs = options['n_rand_dirs']
            theta = lambda f: f/np.cos(f)
            f = lambda dim: np.random.rand(dim)*2-1
            directions = np.array([theta(f(dim)) for i in range(n_rand_dirs)])

        else : 
            if len(sol.gen_p)<=1 : # if only original solution exists, max/min directions are chosen
                directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
            else : # Otherwise search in directions normal to faces
                directions = np.array(hull.equations)[:,0:-1]
            # Filter directions for previously serched directions
            directions = filter_directions(directions,directions_searched)

        if len(directions)>0:
            # Start parallelpool of workers
            sol = start_parallel_pool(directions,network,options,sol)
        else :
            print('All directions searched')

        computations += len(directions)
        directions_searched = np.concatenate([directions_searched,directions],axis=0)

        # Saving data to avoid data loss
        sol.save_xlsx(options['data_file'])
        

        # Creating convex hull
        hull_points = sol.sum_vars[options['mga_variables']].values
        try :
            hull = ConvexHull(hull_points)#,qhull_options='Qs C-1e-32')#,qhull_options='A-0.99')
        except Exception as e: 
            print('did not manage to create hull first try')
            print(e)
            try :
                hull = ConvexHull(hull_points,qhull_options='Qx C-1e-32')
            except Exception as e:
                print('did not manage to create hull second try')
                print(e)


        delta_v = hull.volume - old_volume
        old_volume = hull.volume
        epsilon = delta_v/hull.volume
        epsilon_log.append(epsilon)
        print('####### EPSILON ###############')
        print(epsilon)
    print('performed {} computations'.format(computations))
    return sol


def init_dirs():
    # Import options and start timer
    try :
        setup_file = sys.argv[1]
    except :
        setup_file = 'co2_test'
    dir_path = os.path.dirname(os.path.abspath(__file__))+os.sep
    try :
        options = yaml.load(open(dir_path+'setup_files/'+setup_file+'.yml',"r"),Loader=yaml.FullLoader)
    except : 
        options = yaml.load(open(setup_file,"r"),Loader=yaml.FullLoader)

    options['network_path'] = dir_path+'data/networks/'+options['network_name']
    options['data_file'] = dir_path+options['output_file']+'_'+options['network_name']+'_'+str(len(options['mga_variables']))+'D'+'_eta_'+str(options['mga_slack'])+'.xlsx'
    # set temporary directory
    try :
        tmp_dir = '/scratch/' + str(os.environ['SLURM_JOB_ID']) + '/'
    except :
        tmp_dir = dir_path+'tmp/'
    options['tmp_dir'] = tmp_dir

    # Set number of cores 
    options['cpus'] = os.cpu_count()

    return options

def import_network(options,tmp_network=False):
    network = pypsa.Network()
    if tmp_network == True:
        network.import_from_csv_folder(options['tmp_dir']+'network/')
    else : 
        network.import_from_hdf5(options['network_path'])
    network.snapshots = network.snapshots[0:50]
    return network


#%% main routine 
if __name__=='__main__':
    gc.enable()
    logging.disable()
    mp.set_start_method('spawn')
    timer2 = time.time()
    options = init_dirs()
    # Import network

    network = import_network(options)
    #network.consistency_check()
    # Run initial solution
    network,sol = inital_solution(network,options)
    sol.save_xlsx(options['data_file'])
    # Run MGA using parallel
    sol = run_mga(network,sol,options) 
    # Save data
    sol.save_xlsx(options['data_file'])

    print('finished in time {}'.format( time.time()-timer2))

# %%

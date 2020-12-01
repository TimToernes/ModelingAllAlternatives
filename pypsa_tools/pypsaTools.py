import pypsa
import numpy as np
import pyomo.environ as pyomo_env
import logging
import pyomo
from scipy.spatial import ConvexHull,  Delaunay


def import_network(path):
    network = pypsa.Network()
    network.import_from_hdf5(path)
    network.snapshots = network.snapshots[0:2]
    return network

def createNetwork(dim):
    # Method that will create a pypsa network including only
    # the desired number of buses and generators
    network = pypsa.Network()
    network.import_from_hdf5('../data/networks/euro_00')

    bus_list = ['DK', 'SE', 'NO', 'DE', 'PL', 'CZ', 'NL', 'AT', 'CH']
    bus_list = bus_list[:dim]
    # Removing exessive buses
    for bus in network.buses.index:
        if bus not in bus_list:
            network.remove('Bus', name=bus)
            network.remove('Load', bus)
    # removing exessive links
    for link in network.links.index:
        if not (network.links.loc[link].bus0 in bus_list and network.links.loc[link].bus1 in bus_list):
            network.remove('Link', link)
    # Removing generators from not included buses
    for generator in network.generators.index:
        if not network.generators.loc[generator].bus in bus_list:
            network.remove('Generator', generator)
    # Removing solar generators
    for generator in network.generators.index:
        if network.generators.loc[generator].type == 'solar':
            network.remove('Generator', generator)
    # Removing offwind generators
    for generator in network.generators.index:
        if network.generators.loc[generator].name[-7:] == 'offwind':
            network.remove('Generator', generator)

    network.snapshots = network.snapshots[0:10]
  

    return network

def initialSolution(network):
    logging.disable()
    network.lopf(network.snapshots, 
            solver_name='gurobi'),
    # Add CO2 constraint as % of unconstraint emissions
    co2_emission = [constraint.body() for constraint in network.model.global_constraints.values()][0]
    network.remove('GlobalConstraint',"co2_limit")
    target = (1-0.60)*co2_emission
    network.add("GlobalConstraint","co2_limit",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=target)
    # Solve network again now with CO2 constraint
    network.lopf(network.snapshots, 
                solver_name='gurobi')
    old_objective_value = network.model.objective()
    initial_solution = network.generators.p_nom_opt
 
    return network, old_objective_value, initial_solution


def gen_variables_summed(network):
    generators = [gen_p for gen_p in network.model.generator_p_nom]
    types = ['wind','ocgt','olar']
    variables = []
    for i in range(3):
        gen_p_type = [gen_p  for gen_p in generators if gen_p[-4:]==types[i]]
        variables.append(sum([network.model.generator_p_nom[gen_p] for gen_p in gen_p_type])) 
    variables = variables[0:2]
    return variables

def gen_variables_2D(network,options):
    generators = [gen_p for gen_p in network.model.generator_p_nom]
    types = ['ocgt','wind','olar']
    variables = []
    for i in range(3):
        gen_p_type = [gen_p  for gen_p in generators if gen_p[-4:]==types[i]]
        variables.append(sum([network.model.generator_p_nom[gen_p] for gen_p in gen_p_type]))
    slack = 0    
    point = options['point']
    #point = inital_solution
    print(variables[0]== point[0])
    network.model.ocgt_constraint_u = pyomo_env.Constraint(expr=variables[0]== point[0])
    #network.model.ocgt_constraint_l = pyomo_env.Constraint(expr=variables[0]>= inital_solution[0]+1000-slack)
    network.model.wind_constraint_u = pyomo_env.Constraint(expr=variables[1]== point[1])
    #network.model.wind_constraint_l = pyomo_env.Constraint(expr=variables[1]>= inital_solution[1]+1000-slack)
    #network.model.solar_constraint_u = pyomo_env.Constraint(expr=variables[2]== point[2])
    #network.model.solar_constraint_l = pyomo_env.Constraint(expr=variables[2]>= inital_solution[2]+1000-slack)
    """                                                     
    variables = []
    for bus in network.buses.index:
        var = []
        for generator in network.model.generator_p_nom:
            if network.generators.loc[generator].type == 'wind' and network.generators.loc[generator].bus == bus :
                var.append(network.model.generator_p_nom[generator])
        variables.append(sum(var))
        print(sum(var))    """       
    variables = [network.model.generator_p_nom[gen_p] for gen_p in network.model.generator_p_nom]
    return variables

def gen_variables_full_dim(network):
    variables = [network.model.generator_p_nom[gen_p] for gen_p in network.model.generator_p_nom]
    return variables

def direction_search(network, snapshots,options,direction,variables_func,debug=False): #  MGA_slack = 0.05, point=[0,0,0],dim=3,old_objective_value=0):
# Identify the nonzero decision variables that should enter the MGA objective function.
    old_objective_value = options['old_objective_value']
    MGA_slack = 0.1
    variables = variables_func(network)

    objective = 0
    for i in range(len(variables)):
        #print(variables[i])
        objective += direction[i]*variables[i]

    if debug:
        print(objective)
    # Add the new MGA objective function to the model.
    #objective += network.model.objective.expr * 1e-9
    network.model.mga_objective = pyomo_env.Objective(expr=objective)
    # Deactivate the old objective function and activate the MGA objective function.
    network.model.objective.deactivate()
    network.model.mga_objective.activate()
    # Add the MGA slack constraint.
    network.model.mga_constraint = pyomo_env.Constraint(expr=network.model.objective.expr <= 
                                          (1 + MGA_slack) * old_objective_value)

def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0],rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list

def create_interior_points(hull,m):
    #m = 20

    # Generate Delunay triangulation of hull
    try :
        tri = Delaunay(hull.points[hull.vertices])#,qhull_options='Qs')#,qhull_options='A-0.999')
    except : 
        points = np.append(hull.points[hull.vertices],[np.mean(hull.points,axis=0)],axis=0)            
        tri = Delaunay(points,qhull_options='Qs')
    # Distribute number of points based on simplex size 
    tri.volumes = []
    for i in range(len(tri.simplices)):
        try :
            tri.volumes.append(ConvexHull(tri.points[tri.simplices[i,:]]).volume)
        except : 
            tri.volumes.append(0)
    tri.volumes = np.array(tri.volumes)
    tri.volumes_norm = tri.volumes/sum(tri.volumes)

    tri.n_points_in_tri =  (tri.volumes_norm*m).astype(int)
    # Generate interrior points of each simplex
    interrior_points = []
    for i in range(len(tri.simplices)):
        tri_face = tri.points[tri.simplices[i,:]]
        for j in range(tri.n_points_in_tri[i]):
            dim = len(tri.points[0,:])
            rand_list = rand_split(dim+1)

            new_point = sum([face*rand_n for face,rand_n in zip(tri_face,rand_list)])
            interrior_points.append(new_point)
    interrior_points = np.array(interrior_points)
    return interrior_points

def MGA(network,options,variables_func,debug=False):
    logging.disable()
    data_detail_sum = []
    dim=len(variables_func(network))
    old_volume = 0
    epsilon = 1
    try :
        convergence_percent = options['convergence_percent']
    except:
        print("setting convergenc criteria to 1%")
        convergence_percent = 0.01
    
    while True:
        # if only original solution exists, max/min directions are chosen
        if len(data_detail_sum)<=1 : 
            directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
        # Otherwise search in directions normal to faces    
        else : 
            directions = np.array(hull.equations)[:,0:-1]
        print("number of directions {0}".format(len(directions)))

        if len(directions>250):
            directions = directions[0:250]
            
        # Itterate over directions in batch 
        for direction in directions:
            #print(direction)
            network.lopf(network.snapshots,                                 
                            solver_name='gurobi',                                 
                            extra_functionality=lambda network,                                 
                            snapshots: direction_search(network,snapshots,options,direction,variables_func,debug))
            # Add found point to data_detail
            var = []
            for variable in variables_func(network):
                var.append(pyomo.core.expr.current.evaluate_expression(variable))
            data_detail_sum.append(var)
        
        # Compute new convex hull and calculate termination criteria 
        hull = ConvexHull(data_detail_sum,qhull_options='Q12')#,qhull_options='QJ')
        delta_v = hull.volume - old_volume
        old_volume = hull.volume
        epsilon = delta_v/hull.volume
        print('epsilon {0}'.format(epsilon))
        if (not (epsilon > convergence_percent)): 
            break


    return np.array(data_detail_sum)



"""
x_range = np.concatenate([[data_detail_maxmin.diagonal()],[data_detail_maxmin[6:,:].diagonal()]],axis=0).T

def calc_multiplicity_old(P):

    A = np.array([[-1,0],
                [1,0],
                [0,-1],
                [0,1],
                [1,1],
                [-1,-1]])

    b1 = np.array([x_range[0,0],x_range[0,1],x_range[2,0],x_range[2,1],P[1]-x_range[4,0],x_range[4,1]-P[0]])
    b2 = np.array([x_range[1,0],x_range[1,1],x_range[3,0],x_range[3,1],P[0]-x_range[5,0],x_range[5,1]-P[1]])
    try :
        V1 = con2vert(A,b1)
        H1 = ConvexHull(V1).volume
    except : 
        H1 = 1
    try :
        V2 = con2vert(A,b2)
        H2 = ConvexHull(V2).volume
    except : 
        H2 = 1

    return H1*H2
"""

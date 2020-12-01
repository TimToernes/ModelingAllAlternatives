#%%
import yaml

#%%

setup_dir = 'setup_files/'
network_list = ['euro_00_storage','euro_50_storage','euro_80_storage','euro_95_storage']
slack_list = [0.15,0.30,0.45]

options = {
'network_name' : 'euro_50_storage', #'Scandinavia_co2' or 'euro_30'
'solver_options' : {
  'LogToConsole' : 0,
  'method': 2, # 1 = simplex, 2 = barrier
  'threads': 32,
  'logfile': 'solver.log',
  'crossover' : 0,
  'BarConvTol' : 1.e-8,
  'FeasibilityTol' : 1.e-3},  #1e-2 ###1e-6 # [1e-9..1e-2]
'mga_slack' : 0.1,
'mga_slack_type' : 'fixed', # 'fixed' or 'percent'
'baseline_cost' : 1.64620377e+11,
'mga_variables' : ['wind','solar','H2','battery','co2_emission'] , # wind,solar,ocgt,transmission,H2,battery
'mga_convergence_tol' : 0.05,
'output_file' : 'output/prime'}

#%%

for network in network_list:
    for slack in slack_list:
        options['network_name']=network
        options['mga_slack'] = slack
        filename = setup_dir + 'setup_' + network + '_' + str(slack) + '.yml'
        with open(filename, 'w') as outfile:
            yaml.dump(options, outfile, default_flow_style=False)




# %%

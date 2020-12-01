## Readme for the MAA PyPSA project

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains all files nececary for replicating the results in the resarch article "Modeling all alternatives". 

This document contains the following sections  
1) How to produce PyPSA network model files using the PyPSA-eur-30 package 
2) How to run the MAA method on one or more of the PyPSA network models
3) How to replicate the figures from the article 

The anaconda environment needed to run all files in this repository is included in the _environment.yml_ file. 

## 1 Creating PyPSA models using PyPSA-eur-30
The model used in this project is based on the work conducted in the paper _The Benefits of Cooperation in a Highly Renewable European Electricity Network_  [_doi:10.1016/j.energy.2017.06.004_](https://doi.org/10.1016/j.energy.2017.06.004). All files needed to replicate the model from this article is included in the folder _/model_euro_30/_ also containing a README file for that specific project. 
To replicate the results from the "Modeling all alternatives" paper run the file "model_euro_30/opt_ws_network.py" using the config file "options_MGA_storage.yml". This wil yield four network files placed in the _model_euro_30/network_csv/_ folder. The files will be saved using the naming convention 'euro_(CO2 reduction in %)_storage". 


## 2 Runing the MAA method 
To run the MAA method, the network files created in step 1, must be coppied to the folder _/data/networks/_. Then the file _MAA.py_ can be run with a given setup file, by specifying the file as input. 

$ python MAA.py setup_euro_00_storage_0.3.yml 

The setup files are located in the folder _/setup_files/_ . 


## 3 Data postprocessing 
All postprocesing and figure generation is performed in the script _post_processing_paper.py_. To create the figures used in the paper (Figure 1 to 5) simply run the post_processing script. 
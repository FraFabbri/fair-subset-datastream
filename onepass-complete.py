

# testing different values of epsion - range [0.01 - 0.05, 0.1 - 0.15 ... 0.5]


        
def run_one_(biglst):
    
    (graphname, network_data, S_size, partition_S, option, alpha, factor) = biglst
    
    
    if partition_S == "ER":
        n_groups = len(network_data.mapping_groups)
        print(S_size, partition_S)
        ki = int(S_size/n_groups)
        output_partition = {cat: ki for cat in network_data.groups}

    if partition_S == "PR":
        output_partition = {cat: 
                            int(S_size*network_data.proportions[cat]) 
                            for cat in network_data.groups}
    k = S_size
    print(output_partition)

    path_ = "out/onepass/" + graphname + "/" + "new-strategy" + "/"
    try:
        os.makedirs(path_)
    except:
        pass            

    alpha = round(alpha,2)
    beta = alpha

    #filenames
    config_onepass = [
        "k" + str(k),"mode" + partition_S,"alpha" + str(alpha), "beta" + str(beta), "opt", option, "factor"+ str(factor)
    ]

    filename = path_ + "-".join(config_onepass) + ".tsv"

    check_files = glob.glob(path_ + "*")

    if filename not in check_files:
        
        print(filename)

        onePass = Algorithms(network_data, filename)
        onePass.output_partition = output_partition
        onePass.k = k
        onePass.fairOnepass(network_data,  get_neighbors, delta_neighbors, alpha, beta, option, factor)


from multiprocessing import Process
from multiprocessing import Pool


from src.algorithms import Algorithms
from src.data import Data
from src.submodular import *
from src.config1 import *
import sys
import time
import os
import glob
import numpy as np


pokec1 = True
pokec2 = False

options = ["buffer-size"]
#option = options[int(sys.argv[1])]



# input
if pokec1:
    
    # properties
    directed=True
    attribute = "gender"
    datatype = "network"    
    graphname = "pokec-gender"    
    filename_nodes = "data/pokec/soc-pokec-profiles-filtered.txt"
    filename_edgelist = "data/pokec/soc-pokec-relationships-filtered.txt"

# input
if pokec2:
    
    # properties
    directed=True
    attribute = "age"
    datatype = "network"    
    graphname = "pokec-age" 
    
    # input
    filename_nodes = "data/pokec/age-nodes.txt"
    filename_edgelist = "data/pokec/age-edges.txt"

    


# process input-data
network_data = Data()
network_data.directed = directed
network_data.threshold = None
network_data.datatype = datatype

network_data.initialize_profiles(filename_nodes, attribute)
network_data.mapping_neighbors(filename_edgelist)

N = len(network_data.mapping_id_to_attributes)
    
list_S_size = [x*10 for x in list_S_size]
#args = []
option = "buffer-size"

for S_size in list_S_size:
    for partition_S in ["ER", "PR"]:
        for factor in [2,4]:
            one_arg = [graphname, network_data, S_size, partition_S, option, alpha, factor]
            run_one_(one_arg)
        
        



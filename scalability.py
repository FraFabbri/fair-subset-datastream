


        
def run_scalability(biglst):
    
    (dataname, S_size, partition_S, algoname, epsilon, passes, alpha, beta, option, factor, q) = biglst
    
    
    if partition_S == "ER":
        n_groups = len(network_data.mapping_groups)
        print(S_size, partition_S)
        ki = int(np.ceil(S_size/n_groups))
        output_partition = {cat: ki for cat in network_data.groups}


    if partition_S == "PR":
        output_partition = {cat: 
                            int(np.ceil(S_size*network_data.proportions[cat]))
                            for cat in network_data.groups}
    k = S_size
    print(output_partition)

    path_ = "out/scalability/" + algoname + "/"
    try:
        os.makedirs(path_)
    except:
        pass            
    
    if "MP" in algoname:
        epsilon = round(epsilon,2)

        #filenames
        config_ = [dataname,
            "k" + str(k),"mode" + partition_S,"epsilon" + str(epsilon)]
        
    if "fairgreedy" in algoname:
        #filenames
        config_ = [dataname, "k" + str(k),"mode" + partition_S]
        
    if "greedy" == algoname:
        #filenames
        config_ = [dataname, "k" + str(k),"mode" + partition_S]
        
    if "random" in algoname:
        config_ = [dataname, "k" + str(k),"mode" + partition_S]
        
    if "multipassLS" in algoname:
        config_ = [dataname, "k" + str(k),"mode" + partition_S, "passes"+str(passes)]
    
    if "onepass" in algoname:
        config_ = [dataname, "k" + str(k),"mode" + partition_S, "alpha" + str(alpha), "opt"+ str(option), "factor"+ str(factor)]
        
    if "ls+s" in algoname:
        config_ = [dataname, "k" + str(k),"mode" + partition_S, "q" + str(q)]



    filename = path_ + "-".join(config_) + ".tsv"

    check_files = glob.glob(path_ + "*")

    if filename not in check_files:
        print(algoname)
        print(filename)
        algos = Algorithms(network_data, filename)
        algos.output_partition = output_partition
        algos.k = k

        if "MP" in filename:
            algos.fairMultiPass(network_data,  get_neighbors, delta_neighbors, epsilon)

        if "fairgreedy" in algoname:
            algos.runFairgreedy(network_data, get_neighbors, delta_neighbors)

        if "greedy" == algoname:
            algos.runGreedy(network_data,  get_neighbors, delta_neighbors)
            
        if "random"== algoname:
            algos.randomBaseline(network_data, get_neighbors)
            
        if "multipassLS" in algoname:
            algos.multipassStreamLS(network_data, get_neighbors, delta_neighbors, passes)
            
        if "onepass" in algoname:
            algos.fairOnepass(network_data,  get_neighbors, delta_neighbors, alpha, beta, option, factor)
            
        if "ls+s" in algoname:
            algos.streamLS_with_S(network_data,  get_neighbors, delta_neighbors, q)




from src.algorithms import Algorithms
from src.data import Data
from src.submodular import *
from src.config1 import *
import sys
import time
import os
import glob
import numpy as np

# input! # input! # input!
# input! # input! # input!
# input! # input! # input!


#S_size = list_S_size[int(sys.argv[1])]
#partition_S = list_partition_S[int(sys.argv[2])]

lst_graphs = glob.glob("data/synth/*")
lst_graphs = set([x.replace("-nodes.tsv", "").replace("-edges.tsv", "") for x in lst_graphs])
lst_graphs = sorted(lst_graphs)
S_size = 500
option = "buffer"
directed = False
datatype = "network"
attribute = "attribute"

alpha = beta = 0.5
q = 0.1
epsilon = .1 
passes = 10


#args = []
lst_algonames = ["ls+s"]

for fn in lst_graphs:
    if "N" in fn:
        graphname = fn.split("/")[-1][:-1]
        filename_nodes = fn + "-nodes.tsv"
        filename_edgelist = fn + "-edges.tsv"
        # process input-data
        network_data = Data()
        network_data.directed = directed
        network_data.threshold = None
        network_data.datatype = datatype

        network_data.initialize_profiles(filename_nodes, attribute)
        network_data.mapping_neighbors(filename_edgelist)

        N = len(network_data.mapping_id_to_attributes)
        for algoname in lst_algonames:
            for partition_S in ["PR", "ER"]:
                one_arg = [graphname, S_size, partition_S, algoname, epsilon, passes, alpha, beta, option, factor, q]
                run_scalability(one_arg)









# testing different values of epsion - range [0.01 - 0.05, 0.1 - 0.15 ... 0.5]


        
def run_MP_recsys(biglst):
    
    (dataname, S_size, partition_S, algoname, epsilon, passes, alpha, beta, option, factor, q) = biglst
    
    
    if partition_S == "ER":
        n_groups = len(recsys_data.mapping_groups)
        print(S_size, partition_S)
        ki = int(np.ceil(S_size/n_groups))
        output_partition = {cat: ki for cat in recsys_data.groups}


    if partition_S == "PR":
        output_partition = {cat: 
                            int(np.ceil(S_size*recsys_data.proportions[cat]))
                            for cat in recsys_data.groups}
    k = S_size
    print(output_partition)

    path_ = "out/mp-recsys/" + dataname + "/" + algoname + "/"
    try:
        os.makedirs(path_)
    except:
        pass            
    
    if "MP" in algoname:
        epsilon = round(epsilon,2)

        #filenames
        config_ = [algoname,
            "k" + str(k),"mode" + partition_S,"epsilon" + str(epsilon)]
        
    if "fairgreedy" in algoname:
        #filenames
        config_ = [algoname, "k" + str(k),"mode" + partition_S]
        
    if "greedy" == algoname:
        #filenames
        config_ = [algoname, "k" + str(k),"mode" + partition_S]
        
    if "random" in algoname:
        config_ = [algoname, "k" + str(k),"mode" + partition_S]
        
    if "multipassLS" in algoname:
        config_ = [algoname, "k" + str(k),"mode" + partition_S, "passes"+str(passes)]
    
    if "onepass" in algoname:
        if option != "buffer-size":
            config_ = [algoname, "k" + str(k),"mode" + partition_S, "alpha" + str(alpha), "opt"+ str(option)]
        else:
            config_ = [algoname, "k" + str(k),"mode" + partition_S, "alpha" + str(alpha), "opt"+ str(option), "factor" + str(factor)]
        
    if "ls+s" in algoname:
        config_ = [algoname, "k" + str(k),"mode" + partition_S, "q" + str(q)]



    filename = path_ + "-".join(config_) + ".tsv"

    check_files = glob.glob(path_ + "*")

    if filename not in check_files:

        print(filename)
        algos = Algorithms(recsys_data, filename)
        algos.output_partition = output_partition
        algos.k = k

        if "MP" in filename:
            algos.fairMultiPass(recsys_data,  get_f_recsys, delta_recsys, epsilon)

        if "fairgreedy" in algoname:
            algos.runFairgreedy(recsys_data, get_f_recsys, delta_recsys)

        if "greedy" == algoname:
            algos.runGreedy(recsys_data,  get_f_recsys, delta_recsys)
            
        if "random"== algoname:
            algos.randomBaseline(recsys_data, get_f_recsys)
            
        if "multipassLS" in algoname:
            algos.multipassStreamLS(recsys_data, get_f_recsys, delta_recsys, passes)
            
        if "onepass" in algoname:
            algos.fairOnepass(recsys_data,  get_f_recsys, delta_recsys, alpha, beta, option, factor)
            
        if "ls+s" in algoname:
            algos.streamLS_with_S(recsys_data,  get_f_recsys, delta_recsys, q)
            



from src.algorithms_recsys import Algorithms
from src.data import Data
from src.submodular import *
from src.config1 import *
import sys
import time
import os
import glob
import numpy as np


movielens1M = True
movielens10M = False


datatype = "recsys"    

foldername = "data/ml-1m/"


# process input-data
recsys_data = Data()
recsys_data.datatype = datatype

# load data
uid = 1

recsys_data.load_recsys_data(foldername)
recsys_data.initialize_user_recsys(uid)
recsys_data.lambda_ = .75

dataname = "ml-1m"

epsilon = .1 
passes = 10
alpha = beta = .5
q = 0.1


#args = []
lst_algonames = ["onepass", "MP-FSM", "fairgreedy", "greedy", "random", "ls+s", "multipassLS"]

option = ""

print(lst_algonames)
for S_size in list_S_size:
    
    for partition_S in ["ER", "PR"]:
        
        for algoname in lst_algonames:
            
            if algoname == "onepass":
                
                for option in ["buffer", "no-buffer", "buffer-size"]:
                    
                    if "size" in option:
                        
                        for factor in [2,4]:
                            
                            one_arg = [dataname, S_size, partition_S, algoname, epsilon, passes, alpha, beta, option, factor, q]
                            run_MP_recsys(one_arg)
                    else:
                        
                        one_arg = [dataname, S_size, partition_S, algoname, epsilon, passes, alpha, beta, option, factor, q]
                        run_MP_recsys(one_arg)

            else:
                
                one_arg = [dataname, S_size, partition_S, algoname, epsilon, passes, alpha, beta, option, factor,q]
                run_MP_recsys(one_arg)
        





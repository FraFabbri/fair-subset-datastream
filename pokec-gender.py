from src.algorithms import Algorithms
from src.data import Data
from src.submodular import *
from src.config1 import *

import logging
import sys
import time
import os
import glob

list_partition_S = ["PR", "ER"]

optionGreedy = False
optionFairmulti = True
optionMulticomparison = False

epsilon = 0.2

attribute = "gender"
datatype = "network"

### START ### START ### START
preprocessing=False

# input
filename_nodes = "data/pokec/soc-pokec-profiles-filtered.txt"
filename_edgelist = "data/pokec/soc-pokec-relationships-filtered.txt"

directed=True
# process input-data
network_data = Data()
network_data.directed = directed
network_data.threshold = None
network_data.datatype = datatype

network_data.initialize_profiles(filename_nodes, attribute)
network_data.mapping_neighbors(filename_edgelist)

N = len(network_data.mapping_id_to_attributes)



for S_size in list_S_size:
    for partition_S in ["PR", "ER"]:
        print("---")
        print(S_size)
        print(partition_S)
        print("---")
        logger = logging.getLogger()

        if partition_S == "ER":
            n_groups = len(network_data.mapping_groups)
            print(S_size, n_groups)
            ki = int(S_size/n_groups)
            output_partition = {cat: ki for cat in network_data.groups}

        if partition_S == "PR":
            output_partition = {cat:
                                int(round(S_size*network_data.proportions[cat],0))
                                for cat in network_data.groups}

        k = S_size
        print(sorted(network_data.proportions.items()))
        print(sorted(output_partition.items()))



        filename_log = "log/ex1/" + "-".join(
            ["pokec-gender", "k" + str(k), "mode" + partition_S, "eps" + str(epsilon), "p" + str(passes)]
        )
        out_folder = "out/exp1/" + "-".join(
            ["pokec-gender", "k" + str(k), "mode", partition_S, "eps" + str(epsilon), "p" + str(passes)]
        ) + "/"

        try:
            os.makedirs(out_folder)
        except:
            pass


        fhandler = logging.FileHandler(filename=filename_log + ".log",
                                       mode='a'
                                      )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.DEBUG)


        print(filename_log)
        print()





        if optionGreedy:

            # run greedy
            start_time1 = time.time()

            greedy_ = Algorithms(network_data, out_folder + "fairGreedy.tsv")
            greedy_.output_partition = output_partition
            greedy_.runFairgreedy(network_data, get_neighbors, delta_neighbors)

            delta1 = time.time() - start_time1
            delta1 = round(delta1, 5)
            logging.debug("fairGreedy " + " - " + str(delta1) + " - " + str(greedy_.f_S))


            print()

        if optionFairmulti:

            # run multipass
            start_time2 = time.time()

            multipass = Algorithms(network_data, out_folder+"fairMultipass.tsv")
            multipass.output_partition = output_partition
            multipass.k = k
            multipass.fairMultiPass(network_data, get_neighbors, delta_neighbors, epsilon)

            delta2 = time.time() - start_time2
            delta2 = round(delta2, 5)
            logging.debug("fairMultipass" + " - " + str(delta2) + " - " + str(multipass.f_S))

            print()

        if optionMulticomparison:

            # run multicomparison
            start_time3 = time.time()

            multicomparison = Algorithms(network_data, out_folder +"MultiPassStreamLS.tsv")
            multicomparison.output_partition = output_partition
            multicomparison.multipassStreamLS(network_data, get_neighbors, delta_neighbors, passes)

            delta3 = time.time() - start_time3
            delta3 = round(delta3, 5)
            logging.debug("MultiPassStreamLS" + " - " + str(delta3) + " - " + str(multicomparison.f_S))

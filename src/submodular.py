
from intbitset import intbitset
import numpy as np


### delta neighbors ###
### delta neighbors ###
### delta neighbors ###

def delta_neighbors(data, v, f_S, S_neighbors):
    
    """
    delta_neigh = (total of neigh shared in S) - (total of neigh shared in S U {v}) 
    """
    
    #S_neighbors = intbitset(S_neighbors)
    v_neighbors = intbitset(data.neighbors[v])
    
    f_S1 = len(S_neighbors | v_neighbors)
    delta = f_S1 - f_S
    return delta


### neighbors ###
### neighbors ###
### neighbors ###

def get_neighbors(data, S):
    neigh_S = intbitset()
    for n in S:
        neigh_S.update(data.neighbors[n])
    f_S = len(neigh_S)
    return f_S, neigh_S




def delta_recsys(data, mid, f_S, max_vector):
    """
    """
    #print(len(max_vector))
    new_max_vector = max_vector.copy()
    idx = 0
    for max_value in max_vector:
        potential_new_max = data.VxV[mid, idx]
        if potential_new_max > max_vector[idx]:
            #print(potential_new_max, max_value)
            new_max_vector[idx] = potential_new_max
        idx+=1
    
    lambda_ = data.lambda_
    
    f_S1 = (new_max_vector.sum())*lambda_
    
    # relevance
    relevance_mid = data.VxU[mid]
    # delta
    delta_ = f_S1 + (1-lambda_)*relevance_mid - f_S 
    #if delta_ <0:
        #print(max_vector)
        #print(new_max_vector)
        #print(relevance_mid)
        #print(f_S1, f_S)    
        #print()

    
    return delta_


def get_f_recsys(data, S):
    """
    """
    lambda_ = data.lambda_
    mapping_V = {}
    S_idxs = sorted(S)
    sub_matrix1 = data.VxV[:, S_idxs]
    #print(sub_matrix1.shape)
    #idx_S = np.argmax(sub_matrix, axis=0)
    max_vector = np.max(sub_matrix1, axis=1)

    
    out = max_vector.sum()*lambda_
    # relevance
    tot_relevance = (data.VxU[S_idxs]).sum()
    
    out = out + (1-lambda_)*tot_relevance
    
    return out, max_vector 
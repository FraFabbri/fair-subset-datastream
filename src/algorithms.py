
import numpy as np
from numpy.random import choice
import time
import csv
from intbitset import intbitset
import heapq


        


# sampling
def reservoir_sampling(array, k):
    """
    
    example:
    
    array = list(range(100))
    k = 10
    out = reservoir_sampling(array,k)
    """
    array = list(array)
    sample = array[:k].copy()
    size_left = len(array) - k
    left_array = array[k:]
    idxs = np.random.choice(array, size=size_left, replace=True)
    for position, idx in enumerate(idxs):
        if idx < k:
            sample[idx] = left_array[position]
    return sample





class Algorithms():
    
    """
    """
    def __init__(self, data, tracking_filename):
        
        self.count_k_i = {i:0 for i in data.groups}
        self.tracking_filename = tracking_filename
        
        self.S = set()
        self.f_S = 0
        self.S_neigh = intbitset()
        
        self.track = {}
        self.tracking_delta = {}
        self.tracking_filename = tracking_filename
        
        self.passes = 0


    ####################################################
    ####################################################
    ####################################################
                
    def save_tracking(self):
        """
        """
        with open(self.tracking_filename, "a+") as f:
            writer_ = csv.writer(f, delimiter="\t")
            c = len(self.track)-1
            
            f_S = self.track[c][0]
            unixtime = self.track[c][1]
            readabletime = time.ctime()
            
            onerow = [c, f_S, unixtime, readabletime]
            writer_.writerow(onerow)


    ####################################################
    ####################################################
    ####################################################
            
    

    def fairGreedy(self, data, func, func_delta):
        """
        baseline greedy 
        """
        candidates = []
        
        max_v = -np.inf
        for i in data.groups:
            k_i = self.output_partition[i]
            count_i_elements = self.count_k_i[i]
            if count_i_elements < k_i:
                max_candidate = -np.inf

                for v in data.mapping_groups[i]:
                    if v not in self.tracking_delta:
                        one_delta = func_delta(data, v, self.f_S, self.S_neigh)
                        self.tracking_delta[v] = one_delta
                        
                        # update
                        if one_delta > max_candidate:
                            max_candidate = one_delta
                            v_candidate = v
                    else:
                        if self.tracking_delta[v] > max_candidate:
                            one_delta = func_delta(data, v, self.f_S, self.S_neigh)
                            self.tracking_delta[v] = one_delta
                            
                            # update
                            if one_delta > max_candidate:
                                max_candidate = one_delta
                                v_candidate = v
                
                candidates.append((v_candidate, max_candidate, i))     
        if max_v !=None:
            
            selected_tuple = max(candidates, key=lambda x : x[1])

            # update 
            self.S.update([selected_tuple[0]]) 
            self.f_S, self.S_neigh = func(data, self.S)
            self.count_k_i[selected_tuple[-1]] +=1


    ####################################################
    ####################################################
    ####################################################
                
    
    def runFairgreedy(self, data, func, func_delta):        
        """
        """
        # tracking the improvement
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()

        self.passes = 0
        
        while self.count_k_i != self.output_partition:
            # greedy
            old_f_S = self.f_S
            
            self.fairGreedy(data, func, func_delta)
            
            # tracking the improvement
            track_count = len(self.track)
            self.track[track_count] = (self.f_S, time.time())
            self.save_tracking()
            
            # update-passes
            self.passes+=1
            #print(self.passes)

            if old_f_S == self.f_S:
                break
    ####################################################
    ####################################################
    ####################################################
    

    def fairMultiPass(self, data, func, func_delta, epsilon):
        """
        """
        # tracking the improvement
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()
        
        # max-value
        v_group_max = None
        f_max = 0
        for i in data.groups:
            for v in data.mapping_groups[i]:
                f_v, _ = func(data, set([v]))
                self.tracking_delta[v] = f_v
                
                if f_v > f_max:
                    f_max = f_v
                    v_group_max = (v,i)
        delta_max = f_max
        
        # reservoir-sampling
        sampled_R = {}
        
        for i in data.groups:
            k_i = self.output_partition[i]
            R_i = reservoir_sampling(data.mapping_groups[i], k_i)
        
            sampled_R[i] = R_i
            
        # initialize S
        self.S = set([v_group_max[0]])
        self.count_k_i = {i:0 for i in data.groups}
        self.count_k_i[v_group_max[1]] +=1
        self.f_S, self.S_neigh = func(data, self.S)

        # tracking the improvement
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()

        # initialization parameters
        tau = (1-epsilon)*delta_max
        threshold = delta_max*(epsilon/self.k)
        while tau > threshold:
            
            for i in data.groups:
                k_i = self.output_partition[i]
                for v in data.mapping_groups[i]:
                    count_i_elements = self.count_k_i[i]
                    if v not in self.S and count_i_elements < k_i:
                        
                        ### modify the code HERE ###
                        ### before computing new delta check if old delta il lower than tau!!! ###
                        # THIS SHOULD SAVE TIME #
                        
                        if self.tracking_delta[v] >= tau:
                            one_delta = func_delta(data, v, self.f_S, self.S_neigh)
                            self.tracking_delta[v] = one_delta
                            if one_delta >= tau:

                                # update solution
                                self.S.update([v])
                                self.f_S, self.S_neigh = func(data, self.S)
                                self.count_k_i[i] +=1
                            
            # tracking the improvement
            track_count = len(self.track)
            self.track[track_count] = (self.f_S, time.time())
            self.save_tracking()

            if len(self.S) == self.k:
                break
            
            tau = (1-epsilon)*tau
        
        # post-processing - ensuring fairness
        #question for Yanhao - should we not specify el in R_i != el in S?        
        for i in sampled_R:
            R_i = sampled_R[i]
            for el in R_i:
                count_i_elements = self.count_k_i[i]
                if el not in self.S and count_i_elements < k_i:
                    # update solution
                    self.S.update([el])
                    self.count_k_i[i] +=1
                    
                    
        # final update of the function
        self.f_S, self.S_neigh = func(data, self.S)
        
        # tracking the improvement at FINAL PASS (after post-processing)
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()


    ####################################################
    ####################################################
    ####################################################


    def streamLS(self, data, func, func_delta):
        """
        """
        for i in data.groups:
            for v in data.mapping_groups[i]:
                # check elements

                k_i = self.output_partition[i]                
                count_i_elements = self.count_k_i[i]
                
                if count_i_elements < k_i:
                    # update solution
                    self.S.update([v])
                    self.count_k_i[i] +=1
                else:
                    set_already_selected_i = set([el for el in self.S if el in data.mapping_groups[i]])
                    
                    min_delta_value = +np.inf
                    v_argmin = None
                    
                    for old_el in set_already_selected_i:
                        # check before if it was computed in the previous iterations!
                        if old_el in self.tracking_delta:
                            one_delta = self.tracking_delta[old_el]
                        else:
                            one_delta = func_delta(data, old_el, self.f_S, self.S_neigh)
                            self.tracking_delta[old_el] = one_delta
                            
                        if one_delta < min_delta_value:
                            min_delta_value = one_delta
                            v_argmin = old_el
                   
                    # check
                    if v in self.tracking_delta:
                        if self.tracking_delta[v] >= 2*min_delta_value:
                            
                            delta_potential_el = func_delta(data, v, self.f_S, self.S_neigh)
                            self.tracking_delta[v] = delta_potential_el
                            
                            # check delta
                            if delta_potential_el >= 2*min_delta_value:                        
                                self.S.discard(v_argmin)
                                self.S.update([v])

                    else:
                        delta_potential_el = func_delta(data, v, self.f_S, self.S_neigh)
                        self.tracking_delta[v] = delta_potential_el
                        
                        # check delta
                        if delta_potential_el >= 2*min_delta_value:
                            self.S.discard(v_argmin)
                            self.S.update([v])

                        
        self.f_S, self.S_neigh = func(data, self.S)

        # tracking the improvement
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()


    ####################################################
    ####################################################
    ####################################################
    
    
    def streamLS_with_S(self, data, func, func_delta, q):
        """
        """
        
        # run StreamLS
        # tracking the improvement
        track_count = len(self.track)
        t0 = time.time()
        self.track[track_count] = (0, 0)
        self.save_tracking()

        
        N = len(data.mapping_id_to_attributes)
        q_vector = choice(a=[True, False] , size=N, replace=True, p=[q, 1-q])        
        
        count_q = 0
        for i in data.groups:
            for v in data.mapping_groups[i]:
                # check elements

                k_i = self.output_partition[i]                
                count_i_elements = self.count_k_i[i]
                
                if count_i_elements < k_i:
                    # update solution
                    self.S.update([v])
                    self.count_k_i[i] +=1
                else:
                    one_out_q = q_vector[count_q]
                    count_q += 1
                    
                    if one_out_q:
                        
                        set_already_selected_i = set([el for el in self.S if el in data.mapping_groups[i]])

                        min_delta_value = +np.inf
                        v_argmin = None

                        for old_el in set_already_selected_i:
                            # check before if it was computed in the previous iterations!
                            if old_el in self.tracking_delta:
                                one_delta = self.tracking_delta[old_el]
                            else:
                                one_delta = func_delta(data, old_el, self.f_S, self.S_neigh)
                                self.tracking_delta[old_el] = one_delta

                            if one_delta < min_delta_value:
                                min_delta_value = one_delta
                                v_argmin = old_el

                        # check
                        if v in self.tracking_delta:
                            if self.tracking_delta[v] >= 2*min_delta_value:

                                delta_potential_el = func_delta(data, v, self.f_S, self.S_neigh)
                                self.tracking_delta[v] = delta_potential_el

                                # check delta
                                if delta_potential_el >= 2*min_delta_value:                        
                                    self.S.discard(v_argmin)
                                    self.S.update([v])

                        else:
                            delta_potential_el = func_delta(data, v, self.f_S, self.S_neigh)
                            self.tracking_delta[v] = delta_potential_el

                            # check delta
                            if delta_potential_el >= 2*min_delta_value:
                                self.S.discard(v_argmin)
                                self.S.update([v])

                        
        self.f_S, self.S_neigh = func(data, self.S)

        # tracking the improvement
        
        t1 = time.time() - t0
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, t1)
        self.save_tracking()

    
    

    def multipassStreamLS(self, data, func, func_delta, passes):
        
        # run StreamLS
        # tracking the improvement
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()

        # first run 
        self.streamLS(data, func, func_delta)
        # 1 - passes
        for j in range(2, passes+1):
            for i in data.groups:
                for v in data.mapping_groups[i]:
                    if v not in self.S:
                        group_i_to_compare = data.mapping_groups[i]
                        set_already_selected_i = set([el for el in self.S if el in group_i_to_compare])
                        v_and_deltas = {}
                        
                        min_delta_value = +np.inf
                        v_argmin = None
                        
                        # scroll all elements
                        for old_el in set_already_selected_i:
                            
                            # check before if it was computed in the previous iterations!
                            if old_el in self.tracking_delta:
                                one_delta = self.tracking_delta[old_el]
                            else:
                                one_delta = func_delta(data, old_el, self.f_S, self.S_neigh)
                                self.tracking_delta[old_el] = one_delta

                            if one_delta < min_delta_value:
                                min_delta_value = one_delta
                                v_argmin = old_el

                        # check
                        if v in self.tracking_delta:
                            if self.tracking_delta[v] >= (1. + 1./j)*min_delta_value:

                                delta_potential_el = func_delta(data, v, self.f_S, self.S_neigh)
                                self.tracking_delta[v] = delta_potential_el

                                # check delta
                                if delta_potential_el >= (1. + 1./j)*min_delta_value:                        
                                    self.S.discard(v_argmin)
                                    self.S.update([v])

                        else:
                            delta_potential_el = func_delta(data, v, self.f_S, self.S_neigh)
                            self.tracking_delta[v] = delta_potential_el

                            # check delta
                            if delta_potential_el >= (1. + 1./j)*min_delta_value:
                                self.S.discard(v_argmin)
                                self.S.update([v])
                            
            # update f
            self.f_S, self.S_neigh = func(data, self.S)
            
            # tracking the improvement
            track_count = len(self.track)
            self.track[track_count] = (self.f_S, time.time())
            self.save_tracking()


    ####################################################
    ####################################################
    ####################################################
        
    
    def greedy(self, data, func, func_delta):
        """
        baseline NO FAIR greedy 
        """
        max_delta = -np.inf
        for i in data.groups:
            for v in data.mapping_groups[i]:
                if v not in self.tracking_delta:
                    one_delta = func_delta(data, v, self.f_S, self.S_neigh)
                    self.tracking_delta[v] = one_delta
                    
                    # update
                    if one_delta >= max_delta:
                        max_delta = one_delta
                        max_v = v
                else:
                    if self.tracking_delta[v] > max_delta:
                        one_delta = func_delta(data, v, self.f_S, self.S_neigh)
                        self.tracking_delta[v] = one_delta

                        # update
                        if one_delta >= max_delta:
                            max_delta = one_delta
                            max_v = v


        # update 
        self.S.update([max_v]) 
        self.f_S, self.S_neigh = func(data, self.S)

    
    ####################################################
    ####################################################
    ####################################################
 

    def runGreedy(self, data, func, func_delta):        
        """
        """
        # tracking 
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()

        self.passes = 0
        
        while self.k != len(self.S):
            old_f = self.f_S
            # greedy
            self.greedy(data, func, func_delta)
            
            # tracking the improvement
            track_count = len(self.track)
            self.track[track_count] = (self.f_S, time.time())
            self.save_tracking()
            
            # update-passes
            self.passes+=1
            #print(self.k, self.S)
            if (self.f_S - old_f) == 0:
                break
            #print(self.passes)
        
    ####################################################
    ####################################################
    ####################################################
    


    def randomBaseline(self, data, func):
        """
        """
        # tracking
        track_count = len(self.track)
        self.track[track_count] = (self.f_S, time.time())
        self.save_tracking()
        
        history_f_S = []
        for _ in range(100):
            
            self.S = set()
            for i in data.groups:

                k_i = self.output_partition[i]
                
                pool_candidates = list(data.mapping_groups[i])
                sampled_elements = list(choice(a=pool_candidates, size=k_i, replace=False))
                self.S.update(sampled_elements) 
                
                
            self.f_S, _ = func(data, self.S)
            history_f_S.append(self.f_S)
        
        f_avg = np.mean(history_f_S)
        # tracking the improvement
        track_count = len(self.track)
        self.track[track_count] = (f_avg, time.time())
        self.save_tracking()

        
    ####################################################
    ####################################################
    ####################################################

            
    def generate_T(self, delta_max, LB, alpha):
        """
        """

        new_tau_min = max(delta_max, LB)/(2*self.k)
        new_tau_max = delta_max
        
        a = np.log2(new_tau_min)/np.log2(1+alpha)
        b = np.log2(new_tau_max)/np.log2(1+alpha)

        a = np.ceil(a)
        b = np.floor(b)

        self.tau_min = new_tau_min

        self.tau_max = new_tau_max

        
        if len(self.mapping_T) == 0:

            T = [((1+alpha)**j,j) for j in np.arange(a,b)]
            
            self.mapping_T = {}
            for (tau, j) in T:
                tau = round(tau, 2)
                j = round(j, 2)
                self.mapping_T[j] = tau
   
        
            # update
            self.j_min = round(a,2)
            self.j_max = round(b,2)
            self.mapping_T[self.j_max] = round((1+alpha)**self.j_max, 2)
            self.mapping_T[self.j_min] = round((1+alpha)**self.j_min, 2)
            
            
            self.cardinality_T = len(self.mapping_T)

            
        else:            
            
            old_min = self.j_min
            old_max = self.j_max



            # update
            self.j_min = round(a,2)
            self.j_max = round(b,2)


            self.mapping_T = { round(j,2): self.mapping_T[j] for j in self.mapping_T if j >= self.j_min}

            self.mapping_T[self.j_max] = round((1+alpha)**self.j_max, 2)
            self.mapping_T[self.j_min] = round((1+alpha)**self.j_min, 2)


            if old_max < self.j_max:
                
                for j in np.arange(old_max, self.j_max):
                    self.mapping_T[round(j,2)] = round((1+alpha)**j, 2)
                    
            self.cardinality_T = len(self.mapping_T)
            #print(self.mapping_T)



    ####################################################
    ####################################################
    ####################################################

    
    def greedyTau(self, data, tau, func, func_delta):
        """            
        one-special-fair-greedy 
         - input: tau - S_tau, B, map_R_i
        """
        
        general_candidates = []
        heapq.heapify(general_candidates)
        
        for i in data.groups:
            
            k_i = self.output_partition[i]
            
            count_i_elements = self.count_ki_tau[tau][i]
            # tuple of three
            if count_i_elements < k_i:
                while len(self.tau_selected_candidates[i]) > 0:
                    value_and_v_and_i = heapq.heappop(self.tau_selected_candidates[i])
                    v = value_and_v_and_i[1]
                    if v not in self.all_S_tau[tau]:
                        heapq.heappush(general_candidates, value_and_v_and_i)
                        break

        

        if len(general_candidates) > 0:
        
            # update selected i
            
            selected_tuple =  heapq.heappop(general_candidates)
            #print(selected_tuple)
        
            self.count_ki_tau[tau][selected_tuple[-1]] +=1
            
            # update set tau 
            self.all_S_tau[tau].update([selected_tuple[1]]) 
            
            #self.history_f_S_tau[tau], self.S_neigh_tau[tau]  = func(data, self.all_S_tau[tau])
            return True
        else:
            return False


        
    ####################################################
    ####################################################
    ####################################################

    
    def fairOnepass(self, data, func, func_delta, alpha, beta, option="no-buffer", factor=2):
        """
        """
        
        t0 = time.time()
        # tracking the improvement
        track_count = len(self.track)
        self.track[track_count] = (0, 0)
        self.save_tracking()

        
        
        self.all_S_tau = {}
        self.S_neigh_tau = {}
        
        self.count_ki_tau = {}
        
        self.track_delta_by_tau = {}
        
        self.history_f_S_tau = {}
        
        self.mapping_T = {}
        
        self.history_best_fv = {}
        
        self.B = {i: intbitset() for i in data.groups}
        self.B_size = 0
        
        self.factor = factor
        if option == "buffer-size":
            self.B_limit = factor*self.k
        
        partial_t = time.time()

        delta_max = 0
        LB = 0


        
        # initialize reservoir sampling
        self.map_R_i =  {i: {} for i in data.groups}

        self.sampled_reservoir_sampling_COUNT = {i: 0 for i in data.groups}
        self.sampled_reservoir_sampling = {}


        for i in data.mapping_groups:
            
            Ni = len(data.mapping_groups[i])
            ki = self.output_partition[i]
            
            arr_ = np.random.choice(list(range(Ni)), size= Ni+1, replace=True)
            self.sampled_reservoir_sampling[i] = {idx: value for (idx, value) in enumerate(arr_)}
            
        
        for v in data.mapping_id_to_attributes:
            sol_v = set([v])

            f_v, _ = func(data, sol_v)
            
            self.history_best_fv[v] = f_v
            self.tracking_delta[v] = f_v
            
            if f_v > delta_max:
                delta_max = f_v
            
            i = data.mapping_id_to_attributes[v]
            k_i = self.output_partition[i]

            self.update_reservoir_sampling(v, ki, i)
            
             
            self.generate_T(delta_max, LB, alpha)
            
            
            for j in self.mapping_T:
                tau = self.mapping_T[j]
                
                if tau not in self.all_S_tau:  
                    
                    # create new set and ki counter 
                    self.all_S_tau[tau] = set()
                    
                    self.count_ki_tau[tau] = {ii : 0 for ii in data.groups}
                    
                    self.history_f_S_tau[tau] = 0
                    
                    self.S_neigh_tau[tau] = intbitset()
            
            self.all_S_tau = {
                self.mapping_T[j]: self.all_S_tau[self.mapping_T[j]]
                for j in self.mapping_T
            
            }
            self.S_neigh_tau = {self.mapping_T[j]: self.S_neigh_tau[self.mapping_T[j]] 
                                for j in self.mapping_T}
            
            selected_j = self.j_max
            max_tau = self.mapping_T[selected_j]
            
            #print([(tau,len(self.all_S_tau[tau])) for tau in self.all_S_tau])
            
            while selected_j > self.j_min:
                max_tau = self.mapping_T[selected_j]

                count_ki_single_tau = self.count_ki_tau[max_tau][i]
                
                updated_1 = False
                updated_2 = False
                
                if count_ki_single_tau < k_i:
                    
                    # SAME-OLD STORY = track for each tau old deltaS
                    if self.tracking_delta[v] > max_tau:
                        
                        if self.history_f_S_tau[max_tau] == 0:
                            
                            delta_potential_el = f_v
                            
                            updated_1 = True
                            updated_2 = False
                        
                        else:
                            
                            # check
                            S_tau = self.all_S_tau[max_tau]
                            f_S_tau = self.history_f_S_tau[max_tau] 
                            S_neigh_tau = self.S_neigh_tau[max_tau]
                        

                            delta_potential_el = func_delta(
                                data, v, f_S_tau, S_neigh_tau
                            )
                            self.tracking_delta[v] = delta_potential_el

                            if delta_potential_el >= max_tau:
                                updated_1 = True
                                updated_2 = False
                                
                            else:
                                if delta_potential_el >= beta*LB/self.k:
                                    updated_1 = False
                                    updated_2 = True

                    else:
                        if f_v >= beta*LB/self.k:
                            # check
                            S_tau = self.all_S_tau[max_tau]
                            f_S_tau = self.history_f_S_tau[max_tau] 
                            S_neigh_tau = self.S_neigh_tau[max_tau]
                        

                            delta_potential_el = func_delta(
                                data, v, f_S_tau, S_neigh_tau
                            )
                            self.tracking_delta[v] = delta_potential_el

                            if delta_potential_el >= beta*LB/self.k:
                                updated_1 = False
                                updated_2 = True

               
                if updated_1 == True:
                    
                    # update S_tau
                    self.all_S_tau[max_tau].update([v])
        
                    self.history_f_S_tau[max_tau] += delta_potential_el
                    
                    set_ = intbitset(data.neighbors[v])
                    self.S_neigh_tau[max_tau] = self.S_neigh_tau[max_tau] | set_

                    self.count_ki_tau[max_tau][i] +=1
                    
                    for remaining_tau in self.all_S_tau:
                        if remaining_tau < max_tau:
                            
                            count_ki_single_tau = self.count_ki_tau[remaining_tau][i]
                            if count_ki_single_tau < k_i:
                                
                                # update S_tau
                                self.all_S_tau[remaining_tau].update([v])
                                self.count_ki_tau[remaining_tau][i] +=1
                                
                                if self.history_f_S_tau[remaining_tau] == 0:

                                    self.history_f_S_tau[remaining_tau] = f_v
                                    self.S_neigh_tau[remaining_tau] = set_

                                else:
        
                                    self.history_f_S_tau[remaining_tau] += delta_potential_el                     
                                    self.S_neigh_tau[remaining_tau] = self.S_neigh_tau[remaining_tau] | set_

                                
                    
                    break
                    
                if updated_2 == True:
                    
                    # update-2
                    if option != "no-buffer":
                        self.B[i].update([v])
                        self.B_size +=1
                        


                selected_j = selected_j - 1  
                #print(max_tau)
                
            
            LB = -np.inf
            for tau in self.all_S_tau:
                if tau in self.history_f_S_tau:
                    f_S_tau = self.history_f_S_tau[tau]
                    LB = max(f_S_tau, LB)
                
            #if (v%100000)==0 and v != 0:
                #print(LB, delta_max)
                #print(round(time.time() - partial_t, 0))
                #partial_max_f_S = max(self.history_f_S_tau.values())
                #print(partial_max_f_S)
                #break
                #pass
            


        # tracking the improvement
        t1 = time.time() - t0
        
        track_count = len(self.track)
        
        partial_max_f_S = 0
        partial_max_tau = None

        partial_max_f_S = -np.inf
        for j in self.mapping_T:
            tau = self.mapping_T[j]
            f_S, _ = func(data, self.all_S_tau[tau])

            if f_S > partial_max_f_S:

                partial_max_f_S = f_S
                partial_max_tau = tau

        
        
        self.track[track_count] = (partial_max_f_S, t1)
        self.save_tracking()

        
        
        if option == "no-buffer":

            best_tau = None
            best_f_S = -np.inf
            for tau in self.all_S_tau:
                one_f_S, _ = func(data, self.all_S_tau[tau])
                #print(one_f_S)
                if one_f_S > best_f_S:
                    best_f_S = one_f_S
                    best_tau = tau
            
            #print("first")
            #print(best_f_S)



            for i in data.groups:
                k_i = self.output_partition[i]

                count_ki_single_tau = self.count_ki_tau[best_tau][i]
                
                if count_ki_single_tau < k_i:
                    missing_elements = k_i - count_ki_single_tau

                    lst_new_el = [x for x in self.map_R_i[i].values() if x not in self.all_S_tau[best_tau]][:missing_elements]
                    self.all_S_tau[best_tau].update(lst_new_el)
            
            best_f_S, _ = func(data, self.all_S_tau[best_tau])

           #print("after")
           #print(best_f_S)

            # tracking the improvement
            t2 = time.time() - t0 - t1

            track_count = len(self.track)
            self.track[track_count] = (best_f_S, t2)
            self.save_tracking()

            
                            
        
        if option != "no-buffer":
            #print("done")

            # post-processing
            self.candidates = {}

            if option == "buffer-size":
                B_fv = {}
                for i in self.B:
                    B_fv_group_i = [(v_buffer, self.history_best_fv[v_buffer]) for v_buffer in self.B[i]]
                    
                    B_k_i = self.output_partition[i]*self.factor

                    B_fv_group_i = sorted(B_fv_group_i, key=lambda x: x[1], reverse=True)[:B_k_i]
                    B_fv[i] = B_fv_group_i

                for i in self.B:
                    self.B[i] = intbitset([v for v, value in B_fv[i]])

            for i in data.groups:
                
                
                Ri = intbitset(set(self.map_R_i[i].values()))

                self.candidates[i] =  Ri | self.B[i]

                self.candidates[i] = [(-self.history_best_fv[v], v, i) 
                                                for v in self.candidates[i]]

                print(len(self.candidates[i]))
                
                #heapq.heapify(self.candidates[i])

            potential_tau = []
            for tau in sorted(self.all_S_tau):

                count_ki_single_tau = self.count_ki_tau[tau]

                diff_ = [
                    (count_ki_single_tau[i] - self.output_partition[i]) 
                    for i in count_ki_single_tau]

                if all(x!=0 for x in diff_):
                    #print(diff_)
                    potential_tau.append(tau)
            if potential_tau != []:
                min_incomplete_tau = min(potential_tau)

            else:
                min_incomplete_tau = max(self.all_S_tau.keys())



            best_tau = None
            best_f_S = -np.inf
            for tau in self.all_S_tau:
                one_f_S, _ = func(data, self.all_S_tau[tau])
                #print(one_f_S)
                if one_f_S > best_f_S:
                    #print("change")
                    #print(one_f_S, best_f_S)
                    best_f_S = one_f_S
                    best_tau = tau

            print("best1")
            print(best_f_S)


            # run greedy
            #print(len(selected_subset_tau))
            for tau in self.all_S_tau:
                if tau <= min_incomplete_tau:
                    #print()
                    #print(tau)
                    
                    #print(len(self.tau_selected_candidates))
                    #value, _ = func(data, self.all_S_tau[tau])
                    #print("before")
                    #print(value)
                    while self.k > len(self.all_S_tau[tau]):
                        #fS_int, _ = func(data, self.all_S_tau[tau])
                        #print(fS_int)
                        
                        self.tau_selected_candidates = {i: self.candidates[i].copy() for i in self.candidates}
                        for i in data.groups:
                            heapq.heapify(self.tau_selected_candidates[i]) 

                        #print(sum([len(self.candidates[i]) for i in data.groups]))
                        #old_f = self.history_f_S_tau[tau]
                        # greedy
                        run_ = self.greedyTau(data, tau, func, func_delta)
                        if run_:
                            pass
                        else:
                            break
                    
                    #print()
                    #value, _ = func(data, self.all_S_tau[tau])
                    #print("after")
                    #print(value)

                        #print(len(self.all_S_tau[tau]))
                        #print(self.count_ki_tau[tau])
                        # update-passes
                        
                        #self.passes+=1
                        #if (self.history_f_S_tau[tau] - old_f) == 0:
                        #print(len(self.all_S_tau[tau]))
                        #print()
                        #break

            best_tau = None
            best_f_S = -np.inf
            for tau in self.all_S_tau:
                one_f_S, _ = func(data, self.all_S_tau[tau])
                #print(one_f_S)
                if one_f_S > best_f_S:
                    #print("change")
                    #print(one_f_S, best_f_S)
                    best_f_S = one_f_S
                    best_tau = tau



            for i in data.groups:
                k_i = self.output_partition[i]

                count_ki_single_tau = self.count_ki_tau[best_tau][i]
                
                if count_ki_single_tau < k_i:
                    missing_elements = k_i - count_ki_single_tau

                    lst_new_el = [x for x in self.map_R_i[i].values() if x not in self.all_S_tau[best_tau]][:missing_elements]
                    self.all_S_tau[best_tau].update(lst_new_el)
            

            # tracking the improvement
            t2 = time.time() - t0 - t1
            track_count = len(self.track)
        
            best_f_S, _ = func(data, self.all_S_tau[best_tau])
            #print(best_f_S)
        
            self.track[track_count] = (best_f_S, t2)
            self.save_tracking()

    
    ####################################################
    ####################################################
    ####################################################


        
    def update_reservoir_sampling(self, el, ki, group_i):
        """
        """
        scrolled_idx = self.sampled_reservoir_sampling_COUNT[group_i]
        
        if scrolled_idx < ki:
            
            self.map_R_i[group_i][scrolled_idx] = el
        
        else:

            random_idx = self.sampled_reservoir_sampling[group_i][scrolled_idx]

            
            if random_idx < ki:
                
                self.map_R_i[group_i][random_idx] = el

                
        self.sampled_reservoir_sampling_COUNT[group_i]+=1
        
    

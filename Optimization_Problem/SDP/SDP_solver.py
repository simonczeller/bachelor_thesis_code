import numpy as np
import state_transition
import create_spaces
import SDP_class
from state_transition import fk
import time
import sys
import math

#define cost function
def gk(sdp_instance, sk, x, T):
    if T == sdp_instance.Tmax:
        return sk[-1]*sk[sdp_instance.r_index]
    else:    
        return x*sk[sdp_instance.r_index]

def solve_SDP(sdp_instance):
    Tmax = sdp_instance.Tmax
    states = sdp_instance.S

    #define dictionary of dictionaries with policy and value function for each stage and state
    policy = {k: {s: None for s in states} for k in range(Tmax)}
    value_function = {k: {s: None for s in states} for k in range(Tmax+1)}

    stagestarttime = time.time()
    value_function[Tmax] = {s: gk(sdp_instance, sdp_instance.S[s], 0, Tmax) for s in states}
    print(f"Stage {Tmax} completed in {np.round(time.time()-stagestarttime, 4)}s")

    starttime = time.time()
    
    # Backward induction of SDP (DP-Algorithm)-----------------------------------------------------
    for k in range(Tmax-1, -1, -1):
        stagestarttime = time.time()
        for s_index in states:
            s = states[s_index]
            min_cost = np.inf
            optimal_x = None
            feasibility = False
            for _, x in enumerate(sdp_instance.X):
                #check if x is feasible
                if x > s[-1] or x < s[-1]-sdp_instance.Climit/s[0]:
                    continue
                #calculate expected cost for x
                feasibility = True
                expected_cost = 0
                for e_val, pe in zip(sdp_instance.W[0], sdp_instance.P_e[k]):
                    for a_val, pa in zip(sdp_instance.W[1], sdp_instance.P_a[k]):
                        wk = (e_val, a_val)
                        sk_plus_1 = fk(s_index, x, wk, sdp_instance)
                        expected_cost += gk(sdp_instance, s, x, k) + pe*pa*value_function[k+1][sk_plus_1]
                #set x as optimal if expected cost is lower than the current minimum
                if math.isnan(expected_cost):
                    sys.exit(f"Error: expected_cost is nan in stage {k} for state {s_index} = {s}")

                if expected_cost < min_cost:
                    min_cost = expected_cost
                    optimal_x = x
            
            value_function[k][s_index] = min_cost
            policy[k][s_index] = optimal_x

    return policy, value_function

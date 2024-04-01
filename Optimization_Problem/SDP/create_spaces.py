import numpy as np
import itertools
import scipy.stats as stats
from scipy.stats import norm


def create_statespace(sdp_instance):
    N_s = sdp_instance.N_s
    N_w = sdp_instance.N_w
    p_r = sdp_instance.p_r
    q_r = sdp_instance.q_r
    d_r = sdp_instance.d_r
    r_mins = sdp_instance.r_mins
    r_maxs = sdp_instance.r_maxs
    e_min = sdp_instance.e_min
    e_max = sdp_instance.e_max
    p_D = sdp_instance.p_D
    q_D = sdp_instance.q_D
    d_D = sdp_instance.d_D
    D_mins = sdp_instance.D_mins
    D_maxs = sdp_instance.D_maxs
    a_min = sdp_instance.a_min
    a_max = sdp_instance.a_max
    C_min = sdp_instance.C_min
    C_max = sdp_instance.C_max


    #define random disturbance space---------------------------------------------------------------
    e = np.linspace(e_min, e_max, N_w) 
    a = np.linspace(a_min, a_max, N_w)


    #----------------------------------------------------------------------------------------------

    #define state space----------------------------------------------------------------------------
    #define C
    C = np.linspace(C_min, C_max, N_s)

    #define rho_r and rho_D
    #set bounds for rho_r and rho_D

    delta_r = np.linspace(r_mins[-1], r_maxs[-1], N_s)

    delta_D = np.linspace(D_mins[-1], D_maxs[-1], N_s)

    #define the state space and starting indices
    spacedescr = []
    S = []
    if d_r>0:
        for k in range(d_r):
            S += [np.linspace(r_mins[k], r_maxs[k], N_s)]
            spacedescr.append([r_mins[k], r_maxs[k], N_s])

    S = S + [delta_r for _ in range(p_r)] + [e for _ in range(q_r)]

    for _ in range(p_r):
        spacedescr.append([r_mins[-1], r_maxs[-1], N_s])

    for _ in range(q_r):
        spacedescr.append([e_min, e_max, N_w])
    
    if d_D>0:
        for k in range(d_D):
            S += [np.linspace(D_mins[k], D_maxs[k], N_s)]
    
    S = S + [delta_D for _ in range(p_D)] + [a for _ in range(q_D)] + [C]

    for _ in range(p_D):
        spacedescr.append([D_mins[-1], D_maxs[-1], N_s])

    for _ in range(q_D):
        spacedescr.append([a_min, a_max, N_w])

    spacedescr.append([C_min, C_max, N_s])
    


    # Generate all possible state combinations
    all_combinations = list(itertools.product(*S))

    # Initialize a dictionary to store each state combination with a unique index
    state_dict = {}

    # Iterate over all combinations and assign a unique index
    for index, combination in enumerate(all_combinations):
        state_dict[index] = combination
   

    return state_dict, spacedescr

# Functions to convert between single and multi-dimensional indices of State Space
def get_index(sdp_instance, s):
    spacedescr = sdp_instance.spacedescr
    multi_index = []
    i = 0
    for k in spacedescr:
        index = np.round((s[i] - k[0]) / (k[1] - k[0]) * (k[2]-1))
        if index < 0:
            index = 0
        if index > k[2]-1:
            index = k[2]-1
        multi_index.append(int(index))
        i += 1
    return multi_index

def single_to_multi(index, dimensions):
    multi_index = []
    for dimension in reversed(dimensions):
        multi_index.append(index % dimension)
        index //= dimension
    return list(reversed(multi_index))

def multi_to_single(multi_index, dimensions):
    single_index = 0
    for i, index in enumerate(multi_index):
        product = 1
        for dimension in dimensions[i+1:]:
            product *= dimension
        single_index += index * product
    return single_index


def create_controlspace(sdp_instance):
    N_x = sdp_instance.N_x
    #define control space---------------------------------------------------------------------------
    X = np.linspace(0, sdp_instance.X_max, N_x)
    #-----------------------------------------------------------------------------------------------
    return X

def create_disturbance_space(sdp_instance):
    e_min = sdp_instance.e_min
    e_max = sdp_instance.e_max
    N_w = sdp_instance.N_w
    a_min = sdp_instance.a_min
    a_max = sdp_instance.a_max
    sigma_r = sdp_instance.sigma_r
    sigma_D = sdp_instance.sigma_D
    

    #define random disturbance space---------------------------------------------------------------
    e = np.linspace(e_min, e_max, N_w) 
    a = np.linspace(a_min, a_max, N_w)
    #and their probabilities (vector of length N with N_w probabilities for each entry)
    P_e = [discr_prob(e_min, e_max, N_w, sigma) for sigma in sigma_r]
    P_a = [discr_prob(a_min, a_max, N_w, sigma) for sigma in sigma_D]

    #-----------------------------------------------------------------------------------------------
    return (e, a), P_e, P_a

def discr_prob(min_val, max_val, N, sigma):
    # Create an array of N+1 points from min to max
    points = np.linspace(min_val, max_val, N+1)
    
    # Calculate the midpoint of each interval for calculating probabilities
    midpoints = (points[:-1] + points[1:]) / 2
    
    # Calculate probabilities using the PDF of the normal distribution
    probabilities = stats.norm.pdf(midpoints, 0, sigma)
    
    # Calculate the probabilities for the extreme intervals as the CDF difference
    probabilities[0] = stats.norm.cdf((points[0] + points[1]) / 2, 0, sigma) - stats.norm.cdf(points[0], 0, sigma)
    probabilities[-1] = stats.norm.cdf(points[-1], 0, sigma) - stats.norm.cdf((points[-2] + points[-1]) / 2, 0, sigma)
    
    # Normalize the probabilities so that they sum to 1
    probabilities /= probabilities.sum()
    
    return probabilities

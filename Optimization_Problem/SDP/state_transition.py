import numpy as np
import create_spaces
import sys

def fk(sk_index, xk, wk, sdp_instance):
    s = sdp_instance.S[sk_index]
    #set variables
    D_index = sdp_instance.D_index
    C_index = sdp_instance.C_index
    rho_r_index = sdp_instance.rho_r_index
    rho_D_index = sdp_instance.rho_D_index
    coeff_r = sdp_instance.coeff_r
    c_r = sdp_instance.c_r
    p_r = sdp_instance.p_r
    q_r = sdp_instance.q_r
    d_r = sdp_instance.d_r
    coeff_D = sdp_instance.coeff_D
    c_D = sdp_instance.c_D
    p_D = sdp_instance.p_D
    q_D = sdp_instance.q_D
    d_D = sdp_instance.d_D


    D_old = s[D_index]
    C_old = s[C_index]

    C_new = C_old - xk + D_old 

    rho_r_new = calculate_next_rho(s[rho_r_index:D_index], wk[0], coeff_r, c_r, p_r, q_r)
    rho_D_new = calculate_next_rho(s[rho_D_index:C_index], wk[1], coeff_D, c_D, p_D, q_D)

    #Initialize sk_plus_1
    sk_plus_1 = []

    if d_r>0:
        sk_plus_1 = [s[0]+rho_r_new[0]]
        for k in range(1,d_r):
            sk_plus_1 += [s[k] + s[k-1]]
    
    sk_plus_1 = sk_plus_1 + rho_r_new
    
    if d_D>0:
        sk_plus_1 = [s[D_index]+rho_D_new[0]]
        for k in range(D_index+1,D_index+d_r):
            sk_plus_1 += [s[+k] + s[k-1]]
    
    sk_plus_1 = sk_plus_1 + rho_D_new + [C_new]

    #mapping sk_plus_1 to the state space
    sk_plus_1 = np.array(sk_plus_1)
    sk_plus_1 = create_spaces.get_index(sdp_instance, sk_plus_1)

    for i in range(len(sk_plus_1)):
        if sk_plus_1[i] < 0:
            sk_plus_1[i] = 0
        if sk_plus_1[i] >= sdp_instance.dimensions[i]:
            sk_plus_1[i] = sdp_instance.dimensions[i]-1

    return create_spaces.multi_to_single(sk_plus_1, sdp_instance.dimensions)


def calculate_next_rho(rho_t, error_plus1, coeff, c, p, q):
    index_shift = p

    # Initialize Delta r_t+1 with the constant term and w_t+1
    Delta_t_plus_1 = c + error_plus1

    # Calculate the AR part
    if p > 0:
        for i in range(p):
            Delta_t_plus_1 += coeff[i] * rho_t[i] 

    # Calculate the MA part
    if q > 0:
        for j in range(q):
            error_index = index_shift + j
            Delta_t_plus_1 += coeff[error_index] * rho_t[error_index] 

    Delta_t_plus_1 = np.array([Delta_t_plus_1])
    if p > 0:
        rho_t_plus_1 = np.concatenate([Delta_t_plus_1, rho_t[:p-1]])
    error_plus1 = np.array([error_plus1])
    if q > 0:
        rho_t_plus_1 = np.concatenate([rho_t_plus_1, error_plus1, rho_t[p:p+q-1]])

    return list(rho_t_plus_1)
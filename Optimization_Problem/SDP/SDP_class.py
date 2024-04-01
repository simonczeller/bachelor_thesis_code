import numpy as np
import create_spaces
from SDP_solver import solve_SDP

class SDP_optimization_problem:
     def __init__(self, 
                  p_r, q_r, coeff_r, d_r, c_r, 
                  p_D, q_D, coeff_D, d_D, c_D,  
                  r_mins, r_maxs, D_mins, D_maxs, 
                  e_min, e_max, a_min, a_max,
                  Climit, sigma_r, sigma_D, 
                  Tmax=10, N_s=5, N_x=10, N_w=5, 
                  C_max=100, X_min=0, X_max=50):
      
      #define ARMA parameters
      self.p_r, self.q_r, self.d_r, self.coeff_r, self.c_r = p_r, q_r, d_r, coeff_r, c_r
      self.p_D, self.q_D, self.d_D, self.coeff_D, self.c_D = p_D, q_D, d_D, coeff_D, c_D
      
      #set bounds for the state space
      self.e_min, self.e_max = e_min, e_max
      self.a_min, self.a_max = a_min, a_max
      self.r_mins, self.r_maxs = r_mins, r_maxs
      self.D_mins, self.D_maxs = D_mins, D_maxs
      self.C_max = C_max 
      self.C_min = 0
      self.X_min, self.X_max = X_min, X_max
      self.N_s, self.N_x, self.N_w = N_s, N_x, N_w
      
      #set horizon
      self.Tmax = Tmax
      
      #set limit for C
      self.Climit = Climit

      #define state space
      self.S, self.spacedescr = create_spaces.create_statespace(self)
      
      #indices are used for an easier access to the different parts of the state vector
      self.r_index     = 0
      self.rho_r_index = d_r
      self.D_index     = d_r + p_r + q_r
      self.rho_D_index = d_r + p_r + q_r + d_D
      self.C_index     = -1

      self.dimensions = [N_s for _ in range(d_r + p_r)] + [N_w for _ in range(q_r)] + [N_s for _ in range(d_D + p_D)] + [N_w for _ in range(q_D)] + [N_s]

      #define control space
      self.X = create_spaces.create_controlspace(self)

      #generate sigma for the disturbance space
      self.sigma_r = sigma_r
      self.sigma_D = sigma_D

      #define random disturbance space
      self.W, self.P_e, self.P_a = create_spaces.create_disturbance_space(self)
    
     def solve(self):
        self.policy, self.value_function = solve_SDP(self)
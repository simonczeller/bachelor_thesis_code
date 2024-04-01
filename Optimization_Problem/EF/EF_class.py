import gurobipy as gp
from gurobipy import GRB
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from Functions import discr_prob
from Functions import get_ARIMA_values
import time

class EF_optimization_problem:
    def __init__(self, 
                  p_r, q_r, coeff_r, d_r, c_r, sigma_r, past_r, past_errors_r,
                  p_D, q_D, coeff_D, d_D, c_D, sigma_D, past_D, past_errors_D,
                  Tmax, C0, Cmax, N_w, t):

        #define ARMA parameters
        self.p_r, self.q_r, self.d_r, self.coeff_r, self.c_r = p_r, q_r, d_r, coeff_r, c_r
        self.p_D, self.q_D, self.d_D, self.coeff_D, self.c_D = p_D, q_D, d_D, coeff_D, c_D

        self.t = t

        #set past data
        self.past_r = past_r
        self.past_errors_r = past_errors_r
        self.past_D = past_D
        self.past_errors_D = past_errors_D

        #set initial exposure and maximum exposure
        self.C0 = C0
        self.Cmax = Cmax

        #set horizon
        self.Tmax = Tmax

        #create gurobi model
        self.model = gp.Model("Extensive")

        #create probabilities for the disturbances
        self.P_e = []
        self.P_a = [] 
        for i in range(self.Tmax+1):
            self.P_e.append(discr_prob(N_w, sigma_r[i]))
            self.P_a.append(discr_prob(N_w, sigma_D[i]))

        
        def generate_scenarios(self):
            Tmax = self.Tmax
            P_e = self.P_e
            P_a = self.P_a
            # Initialize with base case scenarios
            scenarios = [([], [], 1)]
            
            for t in range(Tmax):
                new_scenarios = []
                for e_prob, e_val in zip(P_e[t][0], P_e[t][1]):
                    for a_prob, a_val in zip(P_a[t][0], P_a[t][1]):
                        for scenario in scenarios:
                            new_e, new_a, scenario_prob = scenario[0] + [e_val], scenario[1] + [a_val], scenario[2] * e_prob * a_prob
                            new_scenarios.append((new_e, new_a, scenario_prob))
                scenarios = new_scenarios
            
            return scenarios
        
        self.scenarios = generate_scenarios(self)

        S = len(self.scenarios)


        r0 = self.past_r[-1] 

        self.x_0 = self.model.addVar(name="x_0", vtype=GRB.CONTINUOUS)
        self.x = self.model.addVars(S, range(1, Tmax+1), name="x", vtype=GRB.CONTINUOUS)



        def get_r(self, s, t):
            return get_ARIMA_values(self.p_r, self.q_r, self.d_r, self.coeff_r, self.c_r, self.Tmax, self.past_r, self.past_errors_r, s[0])[t]

        def get_D(self, s, t):
            return get_ARIMA_values(self.p_D, self.q_D, self.d_D, self.coeff_D, self.c_D, self.Tmax, self.past_D, self.past_errors_D, s[1])[t]
        

        # Initialize an empty list to collect terms for the objective function
        objective_terms = []

        # Iterate over each scenario and time step to construct the objective function
        for s in range(S):
            for t in range(1, Tmax+1):
                # Ensure get_r returns a numeric value
                r_value = get_r(self, self.scenarios[s], t)
                # Add the term for this scenario and time step
                objective_terms.append(self.scenarios[s][2] * self.x[s, t] * r_value)

        # Sum the initial part of the objective and add the sum of all terms collected
        total_objective = r0 * self.x_0 + gp.quicksum(objective_terms)

        # Set the total objective as the model's objective
        self.model.setObjective(total_objective, GRB.MINIMIZE)

        #add constraints
        # Add exposure limit constraints
        for T in range(self.Tmax+1):
            if (self.t+T)%32 != 0:
                continue
            self.model.addConstrs(C0 - self.x_0 + gp.quicksum((get_D(self, self.scenarios[s], t) - self.x[s, t]) for t in range(1,T+1)) <= Cmax/get_r(self, self.scenarios[s], T) for s in range(S))
        #print("Runtime for adding exposure limit constraints: ", time.time()-start)

        #add positive exposure constraints
        self.model.addConstrs(C0 - self.x_0 >= 0)
        for T in range(self.Tmax+1):
            self.model.addConstrs(C0 - self.x_0 + gp.quicksum((get_D(self, self.scenarios[s], t) - self.x[s, t]) for t in range(1,T+1)) >= 0 for s in range(S))

        #add final exposure to 0 constraint
        self.model.addConstrs(C0 - self.x_0 + gp.quicksum((get_D(self, self.scenarios[s], t) - self.x[s, t]) for t in range(1,Tmax+1)) == 0 for s in range(S))

        #add non-anticipativity constraints
        # Iterate over all pairs of scenarios
        for s in range(S):
            for tilde_s in range(S):
                if s != tilde_s:  # No need to compare a scenario with itself
                    # Get the paths for a and e for both scenarios up to time t
                    for t in range(2,Tmax+1):
                        if self.scenarios[s][0][:t] == self.scenarios[tilde_s][0][:t] and self.scenarios[s][1][:t] == self.scenarios[tilde_s][1][:t]:
                                            # Add non-anticipativity constraints for time t
                                            self.model.addConstr(self.x[s, t-1] == self.x[tilde_s, t-1],
                                                            f"non_anticipativity_{s}_{tilde_s}_{t-1}")

    def solve(self):
        sys.stdout = open(os.devnull, 'w') 
        self.model.optimize()
        sys.stdout = sys.__stdout__
        if self.model.status == GRB.Status.OPTIMAL:
            self.optimal_x = self.x_0.X
        else:
            self.optimal_x = None
        
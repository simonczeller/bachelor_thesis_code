import gurobipy as gp
from gurobipy import GRB
from ARIMA_Forecast import get_ARIMA_forecast
import sys, os
import matplotlib.pyplot as plt
import numpy as np

class CEC_optimization_problem:
   def __init__(self, past_r, past_D, past_errors_r, past_errors_D,
                  p_r, ar_coeff_r, d_r, c_r, q_r,
                  p_D, ar_coeff_D, d_D, c_D, q_D,
                  Tmax, C0, Cmax, t):

      #define ARMA parameters
      self.p_r, self.d_r, self.coeff_r, self.c_r, self.q_r = p_r, d_r, ar_coeff_r, c_r, q_r
      self.p_D, self.d_D, self.coeff_D, self.c_D, self.q_D = p_D, d_D, ar_coeff_D, c_D, q_D

      #set past data
      self.past_r, self.past_errors_r = past_r, past_errors_r
      self.past_D, self.past_errors_D = past_D, past_errors_D

      #set initial exposure and maximum exposure
      self.C0 = C0
      self.Cmax = Cmax

      #set horizon
      self.Tmax = Tmax

      #get ARIMA forecasts
      self.r_forecast = get_ARIMA_forecast(self.p_r, self.q_r, self.d_r, self.coeff_r, self.c_r, self.Tmax, self.past_r, self.past_errors_r)
      self.D_forecast = get_ARIMA_forecast(self.p_D, self.q_D, self.d_D, self.coeff_D, self.c_D, self.Tmax, self.past_D, self.past_errors_D)

      #create gurobi model
      self.model = gp.Model("CEC")

      # Initialize the decision variables
      self.x = self.model.addVars(self.Tmax+1, name="x", lb=0)

      # Set the objective function
      self.model.setObjective(gp.quicksum([self.x[t] * self.r_forecast[t] for t in range(self.Tmax+1)]), GRB.MINIMIZE)

      # Add the constraints to the model
      # Add cumulative constraints for each time period
      for T in range(self.Tmax+1):
         if (t+T)%32 != 0:
            continue
         self.model.addConstr(
               self.C0+gp.quicksum((self.D_forecast[t] - self.x[t]) for t in range(T+1)) <= self.Cmax/self.r_forecast[T],
               name=f"Cumulative_Constraint_{T}_max_exposure")

      #positive exposure constraint
      for T in range(self.Tmax+1):
         self.model.addConstr(
            self.C0+gp.quicksum((self.D_forecast[t] - self.x[t]) for t in range(T+1)) >= 0,
            name=f"Cumulative_Constraint_{T}_negative_exposure")
      
      #Add final exposure cost
      self.model.addConstr(self.C0+gp.quicksum([(self.D_forecast[t] - self.x[t]) for t in range(Tmax+1)]) == 0,
            name=f"Cumulative_Constraint_{Tmax}_zero_exposure")
      
         
   # Solve the model
   def solve(self):
      sys.stdout = open(os.devnull, 'w') 
      self.model.optimize()
      sys.stdout = sys.__stdout__
      if self.model.status == GRB.Status.OPTIMAL:
         self.optimal_x = {t: self.x[t].X for t in range(self.Tmax+1)}
      else:
         self.optimal_x = None
        


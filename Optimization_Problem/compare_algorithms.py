import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import sys
sys.path.append('./CEC')
from CEC_class import CEC_optimization_problem
sys.path = sys.path[:-1]
sys.path.append('./SDP')
from SDP_class import SDP_optimization_problem
from create_spaces import get_index, single_to_multi, multi_to_single
sys.path = sys.path[:-1]
sys.path.append('./EF')
from EF_class import EF_optimization_problem
import time
import sys, os
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 10})

import gurobipy as gp
from gurobipy import GRB

import subprocess
caffeinate_process = subprocess.Popen(['caffeinate', '-w', str(subprocess.os.getpid())])
resultspath = '...'

savings_CEC = []
savings_SDP = []
savings_EF = []
savings_LF = []

simulationstarttime = time.time()

for seedindex in range(50):
    print(f"-------------------Simulation {seedindex+1}-------------------")
    #model exchange rate as ARIMA(1,1,1) process
    np.random.seed(123+seedindex)
    n = 5*960
    phi1 = 0.2
    phi2 = 0
    #phi1 = 0.9
    #phi2 = -0.2
    p_r = 1
    #p_r = 2
    theta1 = -0.23
    #theta1 = 0.5
    q_r = 1
    d_r = 1
    c_r = 0
    delta_r = np.zeros(2)
    errors_r = np.random.normal(0,0.0005,n)

    for i in range(2,n):
        delta_r = np.append(delta_r,errors_r[i] + phi1*delta_r[i-1] + phi2*delta_r[i-2] + theta1*errors_r[i-1])

    #print max and min of delta_r
    print(f"Max: {np.max(delta_r)}")
    print(f"Min: {np.min(delta_r)}")

    delta_r[0] = delta_r[0]+0.93

    #integrate series with starting value 0.93 to simulate ARIMA(2,1,1) process
    r = np.cumsum(delta_r) 

    #fit ARIMA(1,1,1) model
    r_model = ARIMA(r[0:4*960], order=(1,1,1))
    r_model_fit = r_model.fit()

    coeff_r = [r_model_fit.params[0], r_model_fit.params[1]]

    sigma_r = r_model_fit.resid.std()
    #-------------------------------------------------------------------------------------------

    #model inflow as AR(2) process----------------------------------------------------------
    #seed + 1 to have random numbers that are independent of the exchange rate
    np.random.seed(124+seedindex)
    n = 5*960
    phi1=0.6
    phi2=0.4
    p_D = 2
    q_D = 0
    d_D = 0
    c_D = 0
    coeff_D = [phi1, phi2]
    D = [10000]
    errors_D = np.random.normal(0,20,n)

    for i in range(1,n):
        D = np.append(D,errors_D[i] + phi1*D[i-1] + phi2*D[i-2])

    #fit AR(2) model
    D_model = ARIMA(D[0:4*960], order=(2,0,0))
    D_model_fit = D_model.fit()
    coeff_D = [D_model_fit.params[1], D_model_fit.params[2]]
    sigma_D = D_model_fit.resid.std()
    c_D = D_model_fit.params[0]

    #-------------------------------------------------------------------------------------------
    
    #Algorithm Comparison-----------------------------------------------------------------------
    C0 = 8000000
    Climit = 10000000




    #total cost of baseline strategy 
    print("--------------------Baseline Strategy--------------------")
    totalexposure = np.round(np.sum(D[4*960:5*960])+C0,2)
    cost_baseline = np.round(np.sum(r[4*960:5*960]*D[4*960:5*960])+C0*r[4*960], 2)

    print(f"Total hedging: {totalexposure:,}$")
    print(f'Total cost:    {cost_baseline:,}€')
    print(f"Average price: {np.round(cost_baseline/totalexposure,4)}€/$")



    #wait and see as lower bound
    print("--------------------Wait and See MIN----------------")
    wsmodel = gp.Model("wait_and_see")
    Tmax = 960

    x = wsmodel.addVars(Tmax, lb=0)
    wsmodel.setObjective(gp.quicksum([r[4*960+t]*x[t] for t in range(Tmax)]), GRB.MINIMIZE)

    for T in range(Tmax):
        if T%32 != 0:
            continue
        wsmodel.addConstr(
            C0+gp.quicksum((D[4*960+t] - x[t]) for t in range(T+1)) <= Climit/r[4*960+T],
            name=f"Cumulative_Constraint_{T}_max_exposure")

    #positive exposure constraint
    for T in range(Tmax):
        wsmodel.addConstr(
        C0+gp.quicksum((D[4*960+t] - x[t]) for t in range(T+1)) >= 0,
        name=f"Cumulative_Constraint_{T}_negative_exposure")

    # Add final exposure cost
    wsmodel.addConstr(C0+gp.quicksum([(D[4*960+t] - x[t]) for t in range(Tmax)]) == 0,
        name=f"Cumulative_Constraint_{Tmax}_zero_exposure")

    sys.stdout = open(os.devnull, 'w') 
    wsmodel.optimize()
    sys.stdout = sys.__stdout__

    cost_ws_min = np.sum([x[t].X*r[4*960+t] for t in range(Tmax)])

    totalhedging = np.sum([x[t].X for t in range(Tmax)])
    print(f"Total heding:  {np.round(totalhedging,2):,}$")
    print(f"Total cost:    {np.round(cost_ws_min,2):,}€")
    print(f"Average price: {np.round(cost_ws_min/totalhedging,4)}€/$")
    print(f"Savings:       {np.round(cost_baseline-cost_ws_min,2):,}€")



    #wait and see as upper bound
    print("--------------------Wait and See MAX----------------")
    wsmaxmodel = gp.Model("wait_and_see")
    Tmax = 960

    x = wsmaxmodel.addVars(Tmax, lb=0)
    wsmaxmodel.setObjective(gp.quicksum([r[4*960+t]*x[t] for t in range(Tmax)]), GRB.MAXIMIZE)

    for T in range(Tmax):
        if T%32 != 0:
            continue
        wsmaxmodel.addConstr(
            C0+gp.quicksum((D[4*960+t] - x[t]) for t in range(T+1)) <= Climit/r[4*960+T],
            name=f"Cumulative_Constraint_{T}_max_exposure")

    #positive exposure constraint
    for T in range(Tmax):
        wsmaxmodel.addConstr(
        C0+gp.quicksum((D[4*960+t] - x[t]) for t in range(T+1)) >= 0,
        name=f"Cumulative_Constraint_{T}_negative_exposure")

    # Add final exposure cost
    wsmaxmodel.addConstr(C0+gp.quicksum([(D[4*960+t] - x[t]) for t in range(Tmax)]) == 0,
        name=f"Cumulative_Constraint_{Tmax}_zero_exposure")

    sys.stdout = open(os.devnull, 'w') 
    wsmaxmodel.optimize()
    sys.stdout = sys.__stdout__

    cost_ws_max = np.sum([x[t].X*r[4*960+t] for t in range(Tmax)])

    totalhedging = np.sum([x[t].X for t in range(Tmax)])
    print(f"Total heding:  {np.round(totalhedging,2):,}$")
    print(f"Total cost:    {np.round(cost_ws_max,2):,}€")
    print(f"Average price: {np.round(cost_ws_max/totalhedging,4)}€/$")
    print(f"Savings:       {np.round(cost_baseline-cost_ws_max,2):,}€")

    
    
    print("--------------------Limited Forecast Strategy-------------------------")
    Ct = C0
    cost_lf = 0
    hedginglf = 0
    lftmax = 10
    for t in range(4*960,5*960-lftmax):
        lfmodel = gp.Model("limited_forecast")
        x = lfmodel.addVars(lftmax, lb=0)
        lfmodel.setObjective(gp.quicksum([r[t+i]*x[i] for i in range(lftmax)]), GRB.MINIMIZE)
        Ct = Ct + D[t]
        #poritive exposure constraint
        for T in range(1,lftmax):
            if T%32 != 0:
                continue
            lfmodel.addConstr(
                Ct+gp.quicksum((D[t+tc] - x[tc]) for tc in range(T+1)) <= Climit/r[t+T],
                name=f"Cumulative_Constraint_{T}_max_exposure")

        #negative exposure constraint
        for T in range(1,lftmax):
            lfmodel.addConstr(
            Ct+gp.quicksum((D[t+tc] - x[tc]) for tc in range(T+1)) >= 0,
            name=f"Cumulative_Constraint_{T}_negative_exposure")

        # Add final exposure cost
        lfmodel.addConstr(Ct+gp.quicksum([(D[t+tc] - x[tc]) for tc in range(lftmax)]) == 0,
            name=f"Cumulative_Constraint_{5}_zero_exposure")

        sys.stdout = open(os.devnull, 'w')
        lfmodel.optimize()
        sys.stdout = sys.__stdout__
        optimal_x = x[0].X 

        hedginglf += optimal_x

        cost_lf += r[t]*optimal_x
        Ct = Ct - optimal_x
    Ct = Ct + D[5*960-lftmax:5*960].sum()
    hedginglf += Ct
    cost_lf += Ct*r[5*960-1]


    print(f"Total heding:  {np.round(hedginglf,2):,}$")
    print(f"Total cost:    {np.round(cost_lf,2):,}€")
    print(f"Savings:       {np.round(cost_baseline-cost_lf,2):,}€")
    savings_LF.append(np.round(cost_baseline-cost_lf,2))
    
    
    
    #total cost of CEC strategy
    print("--------------------CEC Strategy-------------------------")
    cost_CEC = 0
    ar_coeff_r = coeff_r[0:p_r]
    ar_coeff_D = coeff_D[0:p_D]
    hedgingamount = 0
    hedging = []
    totalstarttime = time.time()
    cec = None
    Ct = C0
    for t in range(4*960,5*960):
        starttime = time.time()
        past_r = r[t-2:t+1]
        past_D = D[t-1:t+1]
        past_errors_r = errors_r[t-2:t+1]
        past_errors_D = errors_D[t-1:t+1]
        Tmax = min(5*960-t, 5)
        Ct = Ct + D[t]
        cec = CEC_optimization_problem(past_r, past_D, past_errors_r, past_errors_D,
                                    p_r, coeff_r, d_r, c_r, q_r,
                                    p_D, coeff_D, d_D, c_D, q_D,
                                    Tmax, Ct, Climit, t)
        cec.solve()
        x = cec.optimal_x[0]
        hedging.append(x)
        cost_CEC += r[t]*x
        hedgingamount += x
        Ct = Ct - x
    hedgingamount += Ct
    cost_CEC += Ct*r[5*960-1]

    print(f"Total time:    {time.time()-totalstarttime:.2f}s")

    print(f"Total heding:  {np.round(hedgingamount,2):,}$")
    print(f"Total cost:    {np.round(cost_CEC,2):,}€")
    print(f"Average price: {np.round(cost_CEC/hedgingamount,4)}€/$")
    print(f"Savings:       {np.round(cost_baseline-cost_CEC,2):,}€")

    savings_CEC.append(np.round(cost_baseline-cost_CEC,2))


    print("--------------------SDP Strategy-------------------------")
    cost_SDP = 0
    Ct = C0
    hedgingamount = 0
    hedging = []
    totalstarttime = time.time()
    sdp = None
    r_mins = [r[4*960]-0.02, -0.05]
    r_maxs = [r[4*960]+0.02, 0.05]
    e_min = -0.001
    e_max = 0.001
    D_mins = [D[4*960]-500]
    D_maxs = [D[4*960]+500]
    a_min = -500
    a_max = 500
    N_s = 1
    N_x = 1
    N_w = 1
    X_min = 0
    X_max = 10000000
    Cmax = 10000000
    Tmax = 10
    sigma_r_list = [sigma_r for _ in range(Tmax)]
    sigma_D_list = [sigma_D for _ in range(Tmax)]

    totalstarttime = time.time()

    sdp = SDP_optimization_problem(p_r, q_r, coeff_r, d_r, c_r, 
                                p_D, q_D, coeff_D, d_D, c_D, 
                                r_mins, r_maxs, D_mins, D_maxs, 
                                e_min, e_max, a_min, a_max,
                                Climit, sigma_r_list, sigma_D_list,
                                Tmax=Tmax, N_s=N_s, N_x=N_x, N_w=N_w, 
                                C_max=Cmax, X_min=X_min, X_max=X_max)

    sdp.solve()

    optimal_x = sdp.policy[0]

    for t in range(4*960,5*960):
        #generate current state vector
        Ct = Ct + D[t]
        st = []
        help_r = r[:t+1]
        if d_r>0:
            for k in range(d_r):
                st += [help_r[t]]
                help_r = np.diff(help_r)

        st = st + [help_r[-i] for i in range(p_r)] + [errors_r[t-i] for i in range(q_r)]
        
        help_D = D[:t+1]
        if d_D>0:
            for k in range(d_D):
                st += [help_D[t]]
                help_D = np.diff(help_D)
        
        st = st + [help_D[t-i] for i in range(p_D)] + [errors_D[t-i] for i in range(q_D)] + [Ct]

        st_index = multi_to_single(get_index(sdp, st), sdp.dimensions)

        x = optimal_x[st_index]
        if x > Ct:
            x = Ct
        if x < Ct-sdp.Climit/r[t]:
            x = Ct-sdp.Climit/r[t]

        #print(f"t = {t}, x={x}, C={Ct*r[t]}")
        hedging.append(x)
        cost_SDP += r[t]*x
        hedgingamount += x
        Ct = Ct - x
        #print(f"t = {t}, x={x}, Ct={Ct*r[t]}")

    hedgingamount += Ct
    hedging.append(Ct)
    cost_SDP += Ct*r[5*960-1]

    print(f"Total time:    {time.time()-totalstarttime:.2f}s")
    print(f"Total heding:  {np.round(hedgingamount,2):,}$")
    print(f"Total cost:    {np.round(cost_SDP,2):,}€")
    print(f"Average price: {np.round(cost_SDP/hedgingamount,4)}€/$")
    print(f"Savings:       {np.round(cost_baseline-cost_SDP,2):,}€")
    savings_SDP.append(np.round(cost_baseline-cost_SDP,2))

    print(f"Current Average savings SDP: {np.round(np.mean(savings_SDP),2):,}€")
    

    print("--------------------EF Strategy----------------")
    cost_ef = 0
    hedgingamount = 0
    hedging = []
    totalstarttime = time.time()
    ef = None
    Ct = C0 
    T = 2
    sigma_r_list = [sigma_r for _ in range(T+1)]
    sigma_D_list = [sigma_D for _ in range(T+1)]
    N_w = 3
    N_w = 3
    for t in range(4*960,5*960):
        #print(f"Time for t={t}: {time.time()-totalstarttime:.2f}s")
        starttime = time.time()
        past_r = r[t-2:t+1]
        past_D = D[t-1:t+1]
        past_errors_r = errors_r[t-2:t+1]
        past_errors_D = errors_D[t-1:t+1]
        Tmax = min(5*960-t, T)
        Ct = Ct + D[t]
        ef = EF_optimization_problem(p_r, q_r, coeff_r, d_r, c_r, sigma_r_list, past_r, past_errors_r,
                                    p_D, q_D, coeff_D, d_D, c_D, sigma_D_list, past_D, past_errors_D,
                                    Tmax, Ct, Climit, N_w, t)
        solvestart = time.time()
        ef.solve()
        x = ef.optimal_x
        hedging.append(x)
        cost_ef += r[t]*x
        hedgingamount += x
        Ct = Ct - x
    hedgingamount += Ct
    cost_ef += Ct*r[5*960-1]


    print(f"Total heding:  {np.round(hedgingamount,2):,}$")
    print(f"Total cost:    {np.round(cost_ef,2):,}€")
    print(f"Average price: {np.round(cost_ef/hedgingamount,4)}€/$")
    print(f"Savings:       {np.round(cost_baseline-cost_ef,2):,}€")
    savings_EF.append(np.round(cost_baseline-cost_ef,2))
    
    print(f"Time for simulation {seedindex+1}: {time.strftime('%H:%M:%S', time.gmtime(time.time()-simulationstarttime))}")


print("--------------------Savings-------------------------")
print(f"Average savings SDP: {np.round(np.mean(savings_SDP),2):,}€")
print(f"Average savings CEC: {np.round(np.mean(savings_CEC),2):,}€")
print(f"Average savings EF: {np.round(np.mean(savings_EF),2):,}€")
print(f"Average savings LF: {np.round(np.mean(savings_LF),2):,}€")

print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-simulationstarttime))}")
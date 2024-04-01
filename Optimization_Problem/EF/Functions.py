import numpy as np
from scipy.stats import norm

def discr_prob(N, sigma):
    # Define the range
    min_val = -2 * sigma
    max_val = 2 * sigma

    # Generate points that divide the range into N intervals
    points = np.linspace(min_val, max_val, N+1)
    
    # Calculate midpoints as return points
    if N == 1:
        return [[1], [0]]
    else:
        returnpoints = (points[:-1] + points[1:]) / 2  # Midpoints of intervals

    # Calculate probabilities
    probabilities = np.zeros(N)
    for i in range(N):
        if i == 0:
            # Include the left tail
            probabilities[i] = norm.cdf(points[i+1], 0, sigma)
        elif i == N-1:
            # Include the right tail
            probabilities[i] = 1 - norm.cdf(points[i], 0, sigma)
        else:
            # Middle intervals
            probabilities[i] = norm.cdf(points[i+1], 0, sigma) - norm.cdf(points[i], 0, sigma)

    # Ensure the probabilities are normalized to sum to 1
    probabilities /= probabilities.sum()

    return [list(probabilities), list(returnpoints)]




def get_ARIMA_values(p, q, d, coeff, c, Tmax, past_values, past_errors, scenario_errors):
    original_past_values = np.array(past_values)
    
    # Step 1: Differencing the series
    if d > 0:
        past_delta = np.diff(past_values, n=d)
        values = list(past_delta)  # Initialize scenario values with differenced values
    else:
        values = list(past_values)

    # Initialize errors with past errors
    errors = list(past_errors) 

    # Step 2: Calculation with the ARMA model
    for t in range(Tmax):
        # Autoregressive part
        ar_part = sum(coeff[j] * values[-(j+1)] for j in range(p))

        # Moving Average part
        ma_part = sum(coeff[p+j] * errors[-(j+1)] for j in range(q))

        errors += [scenario_errors[t]]

        # The forecast value is the sum of the AR part, MA part, and constant, plus the scenario error for the forecast step
        value = c + ar_part + ma_part + scenario_errors[t]

        # Update the list of values with the forecasted value
        values.append(value)


    # Step 3: Reverting Differencing if d > 0
    if d > 0:
        # Start with the last original value before differencing as the base for cumulative sum
        reverted_values = [original_past_values[-1]]
        for value in values[len(original_past_values)-1:]:
            reverted_values.append(reverted_values[-1] + value)

        values = reverted_values

    #set negative forecasts to 0
    values = [max(0, value) for value in values]

    # return the last Tmax+1 values
    return values[-(Tmax+1):]
using LinearAlgebra, Statistics, Distributions, DataStructures, ProgressBars

function create_transition_matrix(p_r, q_r, d_r, coeff_r, c_r, p_D, q_D, d_D, coeff_D, c_D, state_dict, spacedescr, dimensions, w)
    # Initialize the transition matrix with zeros
    num_states = length(state_dict)
    T = zeros(num_states, num_states)
    
    for sk_index in ProgressBar(1:num_states)

        sk = state_dict[sk_index]
        
        # Iterate over all provided realizations of random variables in 'w'
        for (wk, wk_prob) in w
            # Calculate the next state based on the current state and wk
            sk_plus_1_index = next_state(sk_index, state_dict, d_r, p_r, q_r, d_D, p_D, q_D, wk, coeff_r, c_r, coeff_D, c_D, spacedescr, dimensions)
            
            # Update the transition matrix
            T[sk_index, sk_plus_1_index] += wk_prob
        end
    end
    
    # Normalize rows of T to ensure they sum to 1
    for i in 1:num_states
        row_sum = sum(T[i, :])
        if row_sum > 0
            T[i, :] .= T[i, :] / row_sum
        end
    end
    
    return T
end

 
# Discrete probability calculation
function discr_prob(min_val, max_val, N, sigma)
    points = range(min_val, max_val, length=N+1)
    midpoints = (points[1:end-1] .+ points[2:end]) ./ 2
    probabilities = pdf.(Normal(0, sigma), midpoints)
    probabilities[1] = cdf(Normal(0, sigma), (points[1] + points[2]) / 2) - cdf(Normal(0, sigma), points[1])
    probabilities[end] = cdf(Normal(0, sigma), points[end]) - cdf(Normal(0, sigma), (points[end-1] + points[end]) / 2)
    probabilities ./= sum(probabilities)
    return probabilities, midpoints
end

# Create Markov states
function create_statespace(N_s, N_w, p_r, q_r, d_r, r_mins, r_maxs, e_min, e_max, p_D, q_D, d_D, D_mins, D_maxs, a_min, a_max)
    e = range(e_min, e_max, length=N_w)
    a = range(a_min, a_max, length=N_w)
    delta_r = range(r_mins[end], r_maxs[end], length=N_s)
    delta_D = range(D_mins[end], D_maxs[end], length=N_s)

    spacedescr = []
    S = []

    if d_r > 0
        for k in 1:d_r
            push!(S, range(r_mins[k], r_maxs[k], length=N_s))
            push!(spacedescr, [r_mins[k], r_maxs[k], N_s])
        end
    end

    S = vcat(S, fill(delta_r, p_r), fill(e, q_r))

    for _ in 1:p_r
        push!(spacedescr, [r_mins[end], r_maxs[end], N_s])
    end

    for _ in 1:q_r
        push!(spacedescr, [e_min, e_max, N_w])
    end

    if d_D > 0
        for k in 1:d_D
            push!(S, range(D_mins[k], D_maxs[k], length=N_s))
        end
    end

    S = vcat(S, fill(delta_D, p_D), fill(a, q_D))

    for _ in 1:p_D
        push!(spacedescr, [D_mins[end], D_maxs[end], N_s])
    end

    for _ in 1:q_D
        push!(spacedescr, [a_min, a_max, N_w])
    end

    all_combinations = collect(Iterators.product(S...))
    state_dict = OrderedDict((index, combination) for (index, combination) in enumerate(all_combinations))

    number_of_states = length(all_combinations)

    return state_dict, spacedescr, number_of_states
end

# Get the index of a state in the state space
function get_index(s, spacedescr)
    multi_index = Int[]
    for (i, si) in enumerate(s)
        k = spacedescr[i]
        step = (k[2] - k[1]) / (k[3] - 1)
        index = round(Int, (si - k[1]) / step) + 1
        push!(multi_index, index)
        if index < 1
            index = 1
        elseif index > k[3]
            index = k[3]
        end
    end
    return Tuple(multi_index)
end

# Convert multi-index to single index
function multi_to_single(multi_index, dimensions)
    index = 0
    stride = 1
    for i in eachindex(multi_index)
        index += (multi_index[i] - 1) * stride
        stride *= dimensions[i]
    end
    return index + 1
end

# Convert single index to multi-index
function single_to_multi(single_index, dimensions)
    multi_index = []
    for dim in reverse(dimensions)
        div, mod = divrem(single_index - 1, dim)
        push!(multi_index, mod + 1)
        single_index = div + 1
    end
    return Tuple(multi_index)
end


# Calculate the next Markov state for given Markov state and realisation of random variables
function next_state(sk_index, S, d_r, p_r, q_r, d_D, p_D, q_D, wk, coeff_r, c_r, coeff_D, c_D, spacedescr, dimensions)
    rho_r_index = d_r + 1
    D_index = d_r + p_r + q_r + 1
    rho_D_index = d_r + p_r + q_r + d_D +1


    s = S[sk_index]


    rho_r_new = calculate_next_rho(s[rho_r_index:D_index], wk[1], coeff_r, c_r, p_r, q_r)
    rho_D_new = calculate_next_rho(s[rho_D_index:end], wk[2], coeff_D, c_D, p_D, q_D)

    sk_plus_1 = []

    if d_r > 0
        push!(sk_plus_1, s[1]+rho_r_new[1])
        for k in 2:d_r
            push!(sk_plus_1, s[k] + s[k-1])
        end
    end


    sk_plus_1 = vcat(sk_plus_1, rho_r_new)


    if d_D > 0
        push!(sk_plus_1, s[D_index]+rho_D_new[1])
        for k in D_index+1:D_index+d_D
            push!(sk_plus_1, s[k] + s[k-1])
        end
    end


    sk_plus_1 = vcat(sk_plus_1, rho_D_new)

    sk_plus_1 = collect(get_index(sk_plus_1, spacedescr))

    for i in eachindex(sk_plus_1)
        if sk_plus_1[i] < 1
            sk_plus_1[i] = 1
        end
        if sk_plus_1[i] > dimensions[i]
            sk_plus_1[i] = dimensions[i]
        end
    end

    sk_plus_1 = Tuple(sk_plus_1)

    return multi_to_single(sk_plus_1, dimensions)
end


# Calculate the next rho of the Markov State
function calculate_next_rho(rho_t, error_plus1, coeff, c, p, q)
    index_shift = p

    # Initialize Delta r_t+1 with the constant term and w_t+1
    Delta_t_plus_1 = c + error_plus1

    # Calculate the AR part
    if p > 0
        for i in 1:p
            Delta_t_plus_1 += coeff[i] * rho_t[i]
        end
    end

    # Calculate the MA part
    if q > 0
        for j in 1:q
            error_index = index_shift + j
            Delta_t_plus_1 += coeff[error_index] * rho_t[error_index]
        end
    end

    Delta_t_plus_1 = [Delta_t_plus_1]
    if p > 0
        rho_t_plus_1 = vcat(Delta_t_plus_1, rho_t[1:p-1]...)
    end
    error_plus1 = [error_plus1]
    if q > 0
        rho_t_plus_1 = vcat(rho_t_plus_1, error_plus1, rho_t[p+1:min(p+q-1, end)]...)
    end
    return rho_t_plus_1
end

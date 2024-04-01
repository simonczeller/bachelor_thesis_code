using SDDP, HiGHS, Test, ProgressBars, StatsBase
include("Functions.jl")

@time begin
############################################################################################
######################################## Parameters ########################################
############################################################################################
C0 = 8000000
exposure_limit = 10000000
N_s = 5
N_w = N_s
p_r = 1
q_r = 1
d_r = 1
current_r = 0.93
mindelta = -0.002
maxdelta = 0.002
tmax = 5
r_mins = [current_r + 5*mindelta, mindelta] 
r_maxs = [current_r + 5*maxdelta, maxdelta]
e_min = -0.001
e_max = 0.001
c_r = 0
coeff_r = [0.2,-0.23]
sigma_e = 0.0005
p_D = 2
q_D = 0
d_D = 0
D_mins = [8000] 
D_maxs = [12000]
coeff_D = [0.6, 0.4]
c_D = 0
a_min = -40
a_max = 40
sigma_a = 20
############################################################################################


############################################################################################
####################################### Markov chain #######################################
############################################################################################
P_e, e = discr_prob(e_min, e_max, N_w, sigma_e)
P_a, a = discr_prob(a_min, a_max, N_w, sigma_a)

@test sum(P_e) ≈ 1.0
@test sum(P_a) ≈ 1.0

w = [((e[i], a[j]), P_e[i] * P_a[j]) for i in 1:N_w for j in 1:N_w]

#test sum of probabilities of w
@test sum([x[2] for x in w]) ≈ 1.0

state_dict, spacedescr, number_of_states = create_statespace(N_s, N_w, p_r, q_r, d_r, r_mins, r_maxs, e_min, e_max, p_D, q_D, d_D, D_mins, D_maxs, a_min, a_max)

@test number_of_states == N_s^d_r * N_s^p_r * N_w^q_r * N_s^d_D * N_s^p_D * N_w^q_D

dimensions = vcat([N_s for _ in 1:(d_r + p_r)], [N_w for _ in 1:q_r], [N_s for _ in 1:(d_D + p_D)], [N_w for _ in 1:q_D])
for i in keys(state_dict)
    @test multi_to_single(get_index(state_dict[i], spacedescr), dimensions) == i
    @test single_to_multi(i, dimensions) == get_index(state_dict[i], spacedescr)
end

transition_matrix = create_transition_matrix(p_r, q_r, d_r, coeff_r, c_r, p_D, q_D, d_D, coeff_D, c_D, state_dict, spacedescr, dimensions, w)

println("Total number of Markov states for discretization $N_s: $number_of_states\n")

same = true

for (i, row) in enumerate(eachrow(transition_matrix))
    # Find indices where the element is not zero
    r = state_dict[i][1]
    indices = findall(x -> x != 0, row)
    for j in indices
        if r != state_dict[j][1]
            global same = false
        end 
    end
end

if same
    println("\033[1;31mDiscretization too low for random disturbance to influence r.\033[0m")
    error("Discretization too low for random disturbance to influence r")
end

@test all(sum(transition_matrix, dims=2) .≈ 1.0)
############################################################################################


############################################################################################
######################################## SDDP model ########################################
############################################################################################
#starting state 
s0 = multi_to_single(get_index((current_r, 0, 0, 10000, 10000), spacedescr), dimensions)

#define Ω
S_r = range(r_mins[1], r_maxs[1], length=N_s)
S_D = range(D_mins[1], D_maxs[1], length=N_s)

Ω = [(r = r_val, D = D_val, limit = exposure_limit/r_val ) for r_val in S_r for D_val in S_D]

# Initialize a dictionary to hold the probabilities arrays for each Markov state
probabilities_dict = Dict()

for s in keys(state_dict)
    r_target = state_dict[s][1] # r value to match
    D_target = state_dict[s][d_r + p_r + q_r + 1] # D value to match

    # Find the index n for which (r, D) matches (r_target, D_target)
    n = findfirst(ω -> ω.r == r_target && ω.D == D_target, Ω)

    # Check if a match was found
    if isnothing(n)
        error("No match found in Ω for key $(s)")
    end

    # Create probabilities array: 0s everywhere except 1 at the n-th position
    probabilities = zeros(length(Ω)) # Initialize with zeros
    probabilities[n] = 1 # Set the n-th position to 1

    # Store the probabilities array in the dictionary
    probabilities_dict[s] = probabilities
end

Tmax = 5

model = SDDP.MarkovianPolicyGraph(
    transition_matrices = Array{Float64,2}[
        transition_matrix[s0,:]',
        [transition_matrix for _ in 1:Tmax-1]... 
    ],
    sense = :Min,
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
) do subproblem, node
    # Unpack the stage and Markov index.
    t, markov_state = node
    # Define the state variable.
    @variable(subproblem, 0 <= currexposure <= exposure_limit/r_mins[1], SDDP.State, initial_value = C0)
    # Define the control variables.
    @variables(subproblem, begin
        hedging >= 0
        D
        r
        limit
    end)
    # Define the constraints
    @constraints(
        subproblem,
        begin
            currexposure.out == currexposure.in + D - hedging
            currexposure.out <= limit
            currexposure.out >= 0
        end
    )
    if t == Tmax # Last stage
        @constraint(
            subproblem,
            currexposure.out==0
        )
    end

    # Note how we can use `markov_state` to dispatch an `if` statement.
    probability = probabilities_dict[markov_state]

    SDDP.parameterize(subproblem, Ω, probability) do ω
        JuMP.fix(D, ω.D)
        JuMP.fix(r, ω.r)
        @stageobjective(
            subproblem,
            ω.r * hedging
        )
    end
end
############################################################################################


############################################################################################
######################################## train SDDP ########################################
############################################################################################
SDDP.train(model, iteration_limit = 100)
############################################################################################

GC.gc()

end
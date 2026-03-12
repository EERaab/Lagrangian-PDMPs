#The acceptance rate for each of our pdmps is (structurally) the same
#(Valid since for each i P_i(t<T) = exp(-∫_[0,T] λ_i(t) dt))
####################################################################################################


"""
    fwd_acceptance(segment::Segment, edge_number::Integer)::Float64

Returns the forward acceptance rate of the edge with number 'edge_number' along 'segment'. 
"""
#Currently a trivial function, due to convenient structural choices, but can in theory become complicated.
function fwd_acceptance(segment::Segment, edge_number::Integer)::Float64
    return segment.forward_rates[edge_number]::Float64
end

"""
    rev_acceptance(segment::Segment, edge_number::Integer)::Float64

Returns the reverse acceptance rate of the edge with number 'edge_number' along 'segment'. 
"""
function rev_acceptance(segment::Segment, edge_number::Integer)::Float64
    return segment.reverse_rates[edge_number]::Float64
end

"""
    select_edge_number(segment::Segment{N})::Int64 where N

Returns an edge number given the path segment 'segment'.

This is a stochastic function, sampling edges according to their probability as prescribed in 'segment'.
"""
function select_edge_number(segment::Segment{N})::Int64 where N
    if N == 1
        return 1
    end
    q = rand()*sum(segment.forward_rates::MVector{N, Float64})
    s = 0.0
    for i ∈ eachindex(segment.forward_rates::MVector{N, Float64})
        s += segment.forward_rates[i]
        if q < s
            return i#The edge: pdmp.graph.edges[state.split_index.x][i]
        end
    end
    #Due to floats we can in fact come through the loop without a return.
    @warn "Float imprecision in edge selection: Sum $s with termination at $q"
    return N 
end
####################################################################################################


"""
    full_density_kernel!(pdmp::PDMP, state::SplitState, evolution_data::EvolutionData, numerics::NumericalParameters)

Returns the density kernel of the the combined measure μ(x,v), where x, v are encoded in 'state' and the target in 'pdmp'.

The measure μ is defined by log μ(x) = log π(x) (where log π is the target) and v|x being 
normal-distributed with mean zero and covariance inv(G(x)), where G(x) is the soft absolute metric at x.
"""
function full_density_kernel!(pdmp::PDMP, state::SplitState, evolution_data::EvolutionData, numerics::NumericalParameters)
    return exp(pdmp.target.log_density(state.position))*auxiliary_kernel!(pdmp, state, evolution_data, numerics)
end


#Because we simulate forward and backward processes by approximating the process 
#one segment at a time in the fwd direction we can only consider transitions out of this segment 
#(rather than transitions in, say, the fwd time-direction).
#Therefore we cannot look at a single transition/edge and compute its forward and backward acceptance at once.
#Rather we have to look at all transitions out of a segment and then adjust the acceptance accordingly.
#This cause a lot of clunkiness unfortunately (reverse edges, initial/final trackers etc)

"""
    new_point!(pdmp::PDMP, state::SplitState, evo_data::EvolutionData, nums::NumericalParameters, max_time::Float64; verbose::Bool = false, reversed_pdmp::Bool = rand(Bool), thresholds = nothing)

In-place fixed-duration simulation of a (split) PDMP 'pdmp' for the state 'state', and returns the associated Metropolis-Hastings acceptance rate for the end state.

# Arguments
- `pdmp::PDMP`: the (split) PDMP to simulate.
- `state::SplitState`: the initial state of the pdmp, i.e. position, velocity and split index (x, v, α).
- `evo_data::EvolutionData`: the evolution data, encoding numerical information about the simulation.
- `nums::NumericalParameters`: the numerical parameters, describing how the simulation is run.
- `max_time::Float64`: the duration of simulation before we return a point.
- `verbose::Bool = false`: allows some optional printing to partially explore simulation process.
- `reversed_pdmp::Bool = rand(Bool)`: whether to simulate the reversal of the input pdmp or not. 
- `thresholds = nothing`: if a list of positive floats is provided, these are used as thresholds for the PDMP flow simulations (described below).

The state is altered in-place throughout the simulation. The function is stochastic in that it by default randomly picks the PDMP or its reversed form.
Moreover the event times are indirectly randomly generated unless a list of Float64 threshold values for forward rate integrals are given in 'thresholds'.
Note that such an array must be longer than the number of events which generally cannot be precomputed. 

See also [`evaluate_flow!`](@ref), [`algorithm`](@ref).
"""
function new_point!(pdmp::PDMP, state::SplitState, evo_data::EvolutionData, nums::NumericalParameters, max_time::Float64; verbose::Bool = false, reversed_pdmp::Bool = rand(Bool), thresholds = nothing)
    #We start our "clock" at t = 0 and terminate at t = max_time
    time = 0.0
    
    #We compute the initial state acceptance and the exponent term
    acceptance = 1.0/full_density_kernel!(pdmp, state, evo_data, nums)
    acceptance_exponent = 0.0

    initial = true
    rev_edge_number = 1

    is_terminal = time > max_time
    verbose ? println("pdmp is reversed: $reversed_pdmp") : nothing
    k = 0
    while !is_terminal
        #Due to us using isomorphisms to cut down on allocations we have a convoluted representation of the segment and vertex.
        vertex = pdmp.graph.vertices[state.split_index.x]
        segment = evo_data.segments[vertex.segment_rate_number]
        dyn = pdmp.graph.dynamics[vertex.dynamic_number]
        reset_segment!(segment)
        if thresholds === nothing
            threshold = -log(rand())
        else
            k += 1
            try
                threshold = thresholds[k]
            catch 
                #Should force threshold to be not a list but a generator!!!!!
                threshold = -log(rand())                
                verbose ? @warn("Threshold list length exceeded, using randomly generated threshold.") : nothing  
            end
        end 

        verbose ? println("Running $dyn with threshold $threshold on state $((state.position, state.auxiliary))") : nothing
        is_terminal = evaluate_flow!(threshold, pdmp, segment, state, evo_data, nums, dyn, max_duration = max_time - time, reversed_pdmp = reversed_pdmp)
        time += segment.time

        verbose ?  println("Segment after evaluation $segment and time $time") : nothing
        acceptance_exponent += segment.forward_rate_integral - segment.reverse_rate_integral
        
        if initial
            initial = false
        else
            #We adjust the acceptance by the acceptance rate of the transition of the reverse edge
            acceptance *= rev_acceptance(segment, rev_edge_number)
        end

        if !is_terminal#time < max_time
            #We enter a new state (possibly stochastically) depending on the end state of the segment
            #Notably we assume here that the reversed pdmp can use the forward graph (i.e. that the two are isomorphic in some sense)
            
            edge_number = select_edge_number(segment)

            acceptance /= fwd_acceptance(segment, edge_number)

            edge = pdmp.graph.edges[state.split_index.x][edge_number]

            verbose ? println("Moving through edge number $edge_number from vertex $(state.split_index.x) to $(edge.target_vertex_number)") : nothing
            rev_edge_number = reversed_edge_number(pdmp.graph, edge)
            transition = pdmp.graph.transitions[edge.transition_number]
            #We adjust the state 
            transition!(state, pdmp, evo_data, nums, edge, transition)
            verbose ? println("Transition through $transition") : nothing
        end
        verbose ? println("Acceptance rate factor $acceptance with exp. $acceptance_exponent") : nothing
        verbose && is_terminal ? println("State is terminal") : nothing
        verbose ? println("----------") : nothing
        if iszero(acceptance)
            return acceptance
        end        
    end

    #In the final state we include the density of the final state in our acceptance, as well as the exponential factor
    acceptance *= full_density_kernel!(pdmp, state, evo_data, nums)*exp(acceptance_exponent)

    #The state has been updated as we've  run this function, and is included in the input. Hence we only need to return the acceptance rate
    return acceptance
end 

"""
    algorithm(pdmp::PDMP, nums::NumericalParameters; max_point_attempts::Integer = 10, max_time::Float64 = 1.0, initial_state::Union{Nothing, SplitState} = nothing, evo_data::EvolutionData = initialize_evolution_data(pdmp, nums))

Simulates a PDMP using Metropolis-Hastings method to iteratively generate random samples from a target distribution defined in 'pdmp'.

# Arguments
- `pdmp::PDMP`: the (split) PDMP to simulate.
- `nums::NumericalParameters`: the numerical parameters, describing how the simulation is run.
- `max_point_attempts::Integer = 10`: the number of times the algorithm attempts to generate a new sample.
- `max_time::Float64 = 1.0`: the duration for which the PDMP is simulated before suggesting a sample.
- `initial_state::Union{Nothing, SplitState} = nothing,`: allows a user to optionally provide an initial state, otherwise one is randomly generated.
- `evo_data::EvolutionData = initialize_evolution_data(pdmp, nums)`: allows the user to pre-allocate evolution data for the simulation.

The state is altered in-place throughout the simulation. The function is stochastic due to the stochastic nature of `new_point!`. 
Moreover, unless an initial state is given, one is randomly generated. 

See also [`evaluate_flow!`](@ref), [`new_point!`](@ref).

"""
function algorithm(pdmp::PDMP, nums::NumericalParameters; max_point_attempts::Integer = 10, max_time::Float64 = 1.0, initial_state::Union{Nothing, SplitState} = nothing, evo_data::EvolutionData = initialize_evolution_data(pdmp, nums))
    k = 0
    samples = Vector{Float64}[]
    acceptances = Float64[]

    # Handle initial state
    if initial_state === nothing
        initial_state = initialize_state!(pdmp, evo_data, nums)
    end
    push!(samples, initial_state.position)

    # Main loop
    while k < max_point_attempts
        #We take the last state position and initialize a new state with a resampled velocity.
        new_state = initialize_state!(pdmp, evo_data, nums, initial_position = copy.(samples[end]))
        
        acceptance = new_point!(pdmp, new_state, evo_data, nums, max_time)
        
        acceptance = min(1., acceptance)
        if isnan(acceptance) #corresponds to fwd_acceptance = 0
            acceptance = 0.
        end
        push!(acceptances, acceptance)

        if rand() < acceptance
            push!(samples, new_state.position)
        else
            push!(samples, copy.(samples[end])) #for the purposes of this algorithm the copy isn't necessary.
        end

        #We only attempt to generate a set number of points - we do not ensure it
        #This is to avoid infinite/long loops for very small acceptance rates.
        k += 1
    end

    return samples, acceptances
end

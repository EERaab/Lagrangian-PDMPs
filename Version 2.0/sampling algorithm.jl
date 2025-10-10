#The acceptance rate for each of our pdmps is (structurally) the same
#(Valid since for each i P_i(t<T) = exp(-∫_[0,T] λ_i(t) dt))
####################################################################################################
function fwd_acceptance(pdmp::PDMP, vertex::MethodVertex, edge_number::Integer)
    return pdmp.graph.segments[vertex.segment_number].forward_rates[edge_number]
end

function rev_acceptance(pdmp::PDMP, vertex::MethodVertex, edge_number::Integer)
    return pdmp.graph.segments[vertex.segment_number].reverse_rates[edge_number]
end

function select_edge(pdmp::PDMP, state::SplitState)
    vertex = pdmp.graph.vertices[state.split_index.x]
    segment = pdmp.graph.segments[vertex.segment_number]
    q = rand()*sum(segment.forward_rates)
    s = 0.0
    for i ∈ eachindex(segment.forward_rates)
        s += segment.forward_rates[i]
        if q ≤ s
            return i#The edge: pdmp.graph.edges[state.split_index.x][i]
        end
    end
end
####################################################################################################


function full_density_kernel!(pdmp::PDMP, state::SplitState, evolution_data::EvolutionData, numerics::NumericalParameters)
    return exp(pdmp.target.log_density(state.position))*auxiliary_kernel!(pdmp, state, evolution_data, numerics)
end


#Because we simulate forward and backward processes by approximating the process 
#one segment at a time in the fwd direction we can only consider transitions out of this segment 
#(rather than transitions in, say, the fwd time-direction).
#Therefore we cannot look at a single transition/edge and compute its forward and backward acceptance at once.
#Rather we have to look at all transitions out of a segment and then adjust the acceptance accordingly.
#This cause a lot of clunkiness unfortunately (reverse edges, initial/final trackers etc)

function new_point!(pdmp::PDMP, state::SplitState, evo_data::EvolutionData, nums::NumericalParameters, max_time::Float64)
    #We start our "clock" at t = 0 and terminate at t = max_time
    time = 0.0

    #We decide whether to follow the PDMP or its reversal
    reversed_pdmp = rand(Bool)
    
    #We compute the initial state acceptance and the exponent term
    acceptance = 1.0/full_density_kernel!(pdmp, state, evo_data, nums)
    acceptance_exponent = 0.0

    initial = true
    rev_edge_number = 1

    while time < max_time
        #Due to us using isomorphisms to cut down on allocations we have a convoluted representation of the segment and vertex.
        vertex = pdmp.graph.vertices[state.split_index.x]
        segment = pdmp.graph.segments[vertex.segment_number]
        reset_segment!(segment)

        #We follow the flow for a time Δt
        #This updates the reverse and forward rates, as well as the corresponding integrals
        evaluate_flow!(pdmp, vertex, state, evo_data, nums, max_duration = max_time - time, reversed_pdmp = reversed_pdmp)
        time += segment.time

        acceptance_exponent += segment.forward_rate_integral - segment.reverse_rate_integral
        

        if initial
            initial = false
        else
            #We adjust the acceptance by the acceptance rate of the transition of the reverse edge
            acceptance *= rev_acceptance(pdmp, vertex, rev_edge_number)
        end

        if time < max_time
            #We enter a new state (possibly stochastically) depending on the end state of the segment
            #Notably we assume here that the reversed pdmp can use the forward graph (i.e. that the two are isomorphic in some sense)
            edge_number = select_edge(pdmp, state)

            acceptance /= fwd_acceptance(pdmp, vertex, edge_number)

            edge = pdmp.graph.edges[state.split_index.x][edge_number]
            
            rev_edge_number = reversed_edge_number(pdmp.graph, edge)
            
            #We adjust the state 
            transition!(state, pdmp, evo_data, nums, edge)
            
        end

        #If the acceptance is Infinite or NaN we should not keep going
        #This happens when either 1) fwd acceptance/rate is zero
        if isnan(acceptance)
            println("NaN-acceptance")
            @show pdmp
            @show vertex
            @show edge_number
            error("NaN-accept")
            #should reject the point
        elseif isinf(acceptance)
            println("Inf-acceptance")
            @show pdmp
            @show vertex
            @show edge_number
            error("Inf-accept")
            #should accept the point if +inf, reject if -inf (though -Inf should be an error)
        end
    end

    #In the final state we include the density of the final state in our acceptance, as well as the exponential factor
    acceptance *= full_density_kernel!(pdmp, state, evo_data, nums)*exp(acceptance_exponent)

    #The state has been updated as we've  run this function, and is included in the input. Hence we only need to return the acceptance rate
    return acceptance
end 

function algorithm(pdmp::PDMP, nums::NumericalParameters; max_point_attempts::Integer = 10, max_time::Float64 = 1.0, initial_state::Union{Nothing, SplitState} = nothing)
    k = 0
    state_list = SplitState[]
    acceptances = Float64[]
    evo_data = initialize_evolution_data(pdmp)

    # Handle initial state
    if initial_state === nothing
        initial_state = initialize_state!(pdmp, evo_data, nums)
    end
    push!(state_list, initial_state)

    # Main loop
    while k < max_point_attempts
        #We take the last state position and initialize a new state with a resampled velocity.
        new_state = initialize_state!(pdmp, evo_data, nums, initial_position = copy.(state_list[end].position))

        acceptance = new_point!(pdmp, new_state, evo_data, nums, max_time)
        acceptance = min(1, acceptance)

        push!(acceptances, acceptance)

        if rand() < acceptance
            push!(state_list, new_state)
        end

        #We only attempt to generate a fixed number of points - we do not ensure it
        #This is to avoid infinite/long loops for very small acceptance rates.
        k += 1
    end

    return state_list, acceptances
end

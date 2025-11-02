#The acceptance rate for each of our pdmps is (structurally) the same
#(Valid since for each i P_i(t<T) = exp(-∫_[0,T] λ_i(t) dt))
####################################################################################################
function fwd_acceptance(segment::Segment, edge_number::Integer)::Float64
    return segment.forward_rates[edge_number]::Float64
end

function rev_acceptance(segment::Segment, edge_number::Integer)::Float64
    return segment.reverse_rates[edge_number]::Float64
end

function select_edge_number(segment::Segment{N})::Int64 where N
    q = rand()*sum(segment.forward_rates::MVector{N, Float64})
    s = 0.0
    for i ∈ eachindex(segment.forward_rates::MVector{N, Float64})
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

function new_point!(pdmp::PDMP, state::SplitState, evo_data::EvolutionData, nums::NumericalParameters, max_time::Float64; verbose = false)
    #We start our "clock" at t = 0 and terminate at t = max_time
    time = 0.0

    #We decide whether to follow the PDMP or its reversal
    reversed_pdmp = rand(Bool)
    #We compute the initial state acceptance and the exponent term
    acceptance = 1.0/full_density_kernel!(pdmp, state, evo_data, nums)
    acceptance_exponent = 0.0

    initial = true
    rev_edge_number = 1

    is_terminal = time > max_time
    verbose ? println("pdmp is reversed: $reversed_pdmp") : nothing

    while !is_terminal
        #Due to us using isomorphisms to cut down on allocations we have a convoluted representation of the segment and vertex.
        vertex = pdmp.graph.vertices[state.split_index.x]
        segment = evo_data.segments[vertex.segment_rate_number]
        dyn = pdmp.graph.dynamics[vertex.dynamic_number]
        reset_segment!(segment)
        threshold = -log(rand())
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
            
            rev_edge_number = reversed_edge_number(pdmp.graph, edge)
            
            transition = pdmp.graph.transitions[edge.transition_number]
            #We adjust the state 
            transition!(state, pdmp, evo_data, nums, edge, transition)
            
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

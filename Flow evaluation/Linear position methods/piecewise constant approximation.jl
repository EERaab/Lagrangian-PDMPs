function update_position!(pdmp::PDMP, segment::Segment{N}, state::SplitState, max_time::Float64, 
    evolution_data::EvolutionData, numerics::NumericalParameters, threshold::Float64, position_method::VTPiecewiseConstant; reversed_pdmp::Bool = false)::Bool where N
    vertex = pdmp.graph.vertices[state.split_index.x]
    dyn = pdmp.graph.dynamics[vertex.dynamic_number]
    #segment = evo_data.segments[vertex.segment_rate_number]
    is_terminal = false
    while !((threshold ≤ segment.forward_rate_integral)||(0 < max_time ≤ segment.time))
        #We update the forward rate.
        fetch_rates!(segment.forward_rates, pdmp, state, evolution_data, numerics, dyn, reversed_pdmp = reversed_pdmp)

        #If the forward integral is getting close to the threshold we terminate the evolution before reaching our current time t + ϵ where ϵ is the stepsize. 
        #We determine the appropriate step length Δt here.
        ΔI = sum(segment.forward_rates)

        threshtime = (threshold - segment.forward_rate_integral)/ΔI
        if max_time > 0
            Δt = min(max_time - segment.time, threshtime, position_method.step_size)
            if position_method.step_size >= max_time - segment.time && max_time - segment.time < threshtime
                is_terminal = true
            end            
        else
            Δt = min(threshtime, position_method.step_size) 
        end
        #We update the forward rate integrals, time and state position.
        segment.time += Δt
        segment.forward_rate_integral += ΔI*Δt
        if reversed_pdmp
            state.position .-= (state.auxiliary .* Δt)
        else
            state.position .+= (state.auxiliary .* Δt)
        end

        #We terminate the process if we've reached a terminal point. Otherwise we keep on looping.
        #Technically this should already be handled by the condition in the while-loop
        if position_method.step_size > Δt
            return is_terminal
        end
    end
    return is_terminal
end

function compute_backward_approximated_integral!(pdmp::PDMP, segment::Segment{N}, state::SplitState, 
    evolution_data::EvolutionData, numerics::NumericalParameters, position_method::VTPiecewiseConstant; reversed_pdmp::Bool = false)::Nothing where N
    vertex = pdmp.graph.vertices[state.split_index.x]
    dyn = pdmp.graph.dynamics[vertex.dynamic_number]
    #segment = evo_data.segments[vertex.segment_rate_number]
    while segment.time > 0
        #We update the reverse rates
        fetch_rates!(segment.reverse_rates, pdmp, state, evolution_data, numerics, dyn, reverse = true, reversed_pdmp = reversed_pdmp)

        #The step size is set to ensure we are not exceeding our termination time.
        Δt = min(segment.time, position_method.step_size)
            
        #We update the reverse rate integral, time and state position.
        segment.time -= Δt
        segment.reverse_rate_integral += sum(segment.reverse_rates)*Δt
        if reversed_pdmp
            state.position .+= (state.auxiliary .* Δt)
        else
            state.position .-= (state.auxiliary .* Δt)
        end
        #We terminate the process if we've reached a terminal point. Otherwise we keep on looping.
        #Technically this should already be handled in the while-loop condition
        if position_method.step_size > Δt   
            return nothing
        end
    end
    return nothing
end


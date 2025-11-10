function update_position!(pdmp::PDMP, segment::Segment{N}, state::SplitState, max_time::Float64, evolution_data::EvolutionData, 
    numerics::NumericalParameters, threshold::Float64, position_method::VTAdaptivePiecewiseConstant; reversed_pdmp::Bool = false)::Bool where N
    #We initialize a trash state that is repeatedly overwritten when computing the adaptive step
    vertex = pdmp.graph.vertices[state.split_index.x]
    dyn = pdmp.graph.dynamics[vertex.dynamic_number]
    
    #OPTIMIZE
    fwd_state = evolution_data.workspace.adaptive_data.adaptive_state
    fwd_state.auxiliary .= state.auxiliary
    fwd_rates = evolution_data.workspace.adaptive_data.adaptive_fwd_rates
    is_terminal = false
    while !((threshold ≤ segment.forward_rate_integral)||(0 < max_time ≤ segment.time))
        #We should update the rate, but we want to adapt to ρ, not λ. Hence we fetch ρ instead. The actual rate(s) is max(0, ρ) (or max(0,ρ_1), max(0, ρ_2),...)
        fetch_rates!(segment.forward_rates, pdmp, state, evolution_data, numerics, dyn, reversed_pdmp = reversed_pdmp)
        
        
        #We introduce convient shorthands
        h = position_method.step_guess
        tol = position_method.tolerance

        #We compute the forward state position
        if reversed_pdmp
            fwd_state.position .= state.position .- (state.auxiliary .* (h/2))
        else
            fwd_state.position .= state.position .+ (state.auxiliary .* (h/2))
        end
        #We compute the forward rate at the forward state. Note that this overwrites the evolution data.
        fetch_rates!(fwd_rates, pdmp, fwd_state, evolution_data, numerics, dyn, reversed_pdmp = reversed_pdmp, adaptive_fwd = true)

        
        #We compute the adapted step size
        rho = evolution_data.workspace.adaptive_data.rho
        fwd_rho = evolution_data.workspace.adaptive_data.fwd_rho
        #M = -Inf
        #@inbounds for i in eachindex(rho)
        #    L =  abs(rho[i]-fwd_rho[i])
        #    if L > M
        #        M = L
        #    end
        #end
        #mult = sqrt(tol/(h*M))
        mult = sqrt(tol/(h*maximum(abs.(rho - fwd_rho))))
        if mult > position_method.max_adaptive_rel_size
            ϵ = position_method.max_adaptive_rel_size*h
        elseif mult < (position_method.max_adaptive_rel_size)^(-1)
            ϵ = h*(position_method.max_adaptive_rel_size)^(-1)
        else
            ϵ = h * mult
        end

        #If the forward integral is getting close to the threshold we terminate the evolution before reaching our current time t + ϵ where ϵ is the stepsize. 
        #We determine the appropriate step length Δt here.
        ΔI = sum(segment.forward_rates)

        threshtime = (threshold - segment.forward_rate_integral)/ΔI
        if max_time > 0
            Δt = min(max_time - segment.time, threshtime, ϵ)
            if ϵ ≥ max_time - segment.time && threshtime ≥ max_time - segment.time  #i.e. Δt = max_time - segment.time
                is_terminal = true
            end   
        else
            Δt = min(threshtime, ϵ) 
        end

        
        #We update the forward rate integral, time and state position.
        segment.time += Δt
        segment.forward_rate_integral += ΔI*Δt
        if reversed_pdmp
            state.position .-= (state.auxiliary.*Δt)
        else
            state.position .+= (state.auxiliary.*Δt)
        end
        
        #We terminate the process if we've reached a terminal point. Otherwise we keep on looping.
        #Technically this should already be handled by the condition in the while-loop
        if ϵ > Δt
            return is_terminal
        end
    end
    return is_terminal
end

function compute_backward_approximated_integral!(pdmp::PDMP, segment::Segment{N}, state::SplitState, 
    evolution_data::EvolutionData, numerics::NumericalParameters, position_method::VTAdaptivePiecewiseConstant; reversed_pdmp::Bool = false)::Nothing where N   #We initialize a trash state that is repeatedly overwritten when computing the adaptive step
    #OPTIMIZE
    vertex = pdmp.graph.vertices[state.split_index.x]
    dyn = pdmp.graph.dynamics[vertex.dynamic_number]
    #segment = evo_data.segments[vertex.segment_rate_number]

    fwd_state = evolution_data.workspace.adaptive_data.adaptive_state
    fwd_state.auxiliary .= state.auxiliary

    fwd_rates = evolution_data.workspace.adaptive_data.adaptive_fwd_rates
    while segment.time > 0
        #We want to determine the adaptive step based on ρ so we fetch ρ instead of the reverse rate λ.
        fetch_rates!(segment.reverse_rates, pdmp, state, evolution_data, numerics, dyn, reverse = true, reversed_pdmp = reversed_pdmp)
        
        #We introduce convient shorthands
        h = position_method.step_guess
        tol = position_method.tolerance

        #We compute the forward state position
        if reversed_pdmp
            fwd_state.position .= state.position .+ (state.auxiliary .*(h/2))
        else
            fwd_state.position .= state.position .- (state.auxiliary .* (h/2))
        end

        #We compute the forward rate at the forward state. Note that this overwrites the evolution data.
        fetch_rates!(fwd_rates, pdmp, fwd_state, evolution_data, numerics, dyn, reverse = true, reversed_pdmp = reversed_pdmp, adaptive_fwd = true)

        #We compute the adapted step size
        rho = evolution_data.workspace.adaptive_data.rho
        fwd_rho = evolution_data.workspace.adaptive_data.fwd_rho
        #M = -Inf
        #@inbounds for i in eachindex(rho)
        #    L =  abs(rho[i]-fwd_rho[i])
        #    if L > M
        #        M = L
        #    end
        #end
        #mult = sqrt(tol/(h*M))
        mult = sqrt(tol/(h*maximum(abs.(rho - fwd_rho))))
        if mult > position_method.max_adaptive_rel_size
            ϵ = position_method.max_adaptive_rel_size*h
        elseif mult < (position_method.max_adaptive_rel_size)^(-1)
            ϵ = h*(position_method.max_adaptive_rel_size)^(-1)
        else
            ϵ = h * mult
        end
        #The step size is set to ensure we are not exceeding our termination time.
        Δt = min(segment.time, ϵ)
            
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
        if ϵ > Δt   
            return nothing
        end
    end
    return nothing
end
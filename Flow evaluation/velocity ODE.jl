function evaluate_flow!(threshold::Float64, pdmp::PDMP, segment::Segment{N}, state::SplitState, evo_data::EvolutionData, 
    numerics::NumericalParameters, dyn::VelocityODE; max_duration::Float64 = 0.0, reversed_pdmp::Bool = false) where N

    #The "parameters" (param) that determine our equation of motion and rates is partially encoded in evo_tensors.
    fetch_evo_data!(pdmp, evo_data, numerics, state, dyn)
        
    reset_integrator!(evo_data, state, threshold, reversed_pdmp)
    integrator = evo_data.integrator
    du = evo_data.long_trash_vector

    #We explicitly assume that the events are of a single type here!
    #In a future implementation this could change.
    
    #### FOR SOME REASON integrator.f.f(du,u0, p, nothing) allocates.
    # We get around this by using velocity_dynamics_split_rho! 
    # (AND ASSUME THAT EVERY CALL TO THIS dyn METHOD defines such a function)
    # Less elegant, but we cannot abide allocations.
    # Old, alloc version: 'integrator.f.f(du, evo_data.velocity_u0, integrator.p, 0.0)'
    velocity_dynamics_split_rho!(du, evo_data.velocity_u0, integrator.p)

    segment.reverse_rates[1] = du[2]

    sol = solve!(integrator)

    get_du!(du, integrator)
    segment.forward_rates[1] = du[1]
    
    final_state = sol.u[end]

    update_velocity_flow_values!(segment, state, final_state)
    
    return false
end

function update_velocity_flow_values!(segment, state, final_state)
    segment.forward_rate_integral = final_state[1]
    segment.reverse_rate_integral = final_state[2] + final_state[3]
    @views state.auxiliary .= final_state[4:end]
    return nothing
end
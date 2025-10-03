include("definitions.jl")

function evaluate_segment!(pdmp::PDMP, state::BinaryState, evolution_data::EvolutionData, numerics::NumericalParameters, dyn_type::PositionVelocity; max_duration::Float64 = 0.0)
    #We pick a stopping threshold randomly.
    threshold = -log(rand())

    #We set up the initial segment.
    #Depending on the PDMP we can have many different rates.    
    segment = PathSegmentValues{typeof(pdmp.method), rate_number(pdmp, dyn_type)}(dyn = dyn_type)

    #We update the state and forward rate and its integral until we reach termination, i.e. the max time or the threshold. A zero max_time means we do not bound time.
    update_position!(pdmp, segment, state, max_duration, evolution_data, numerics, threshold, numerics.position_method)
    forward_i = segment.forward_rate_integral
    #We store the current position. In the backwards iteration it is altered.
    evolution_data.fwd_position .= state.position
    time = segment.time

    #We update the reverse integral and reverse rate for the time.
    compute_backward_approximated_integral!(pdmp, segment, state, evolution_data, numerics, numerics.position_method)
    
    #We restore the end position and time.
    state.position .= evolution_data.fwd_position
    segment.time = time
    if segment.terminal
        return segment, Terminal() 
    end
    new_dyn = select_dynamic(segment)
    return segment, new_dyn
end
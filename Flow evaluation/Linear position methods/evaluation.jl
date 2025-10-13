include("definitions.jl")

function evaluate_flow!(pdmp::PDMP, segment::Segment{N}, state::SplitState, evo_data::EvolutionData, 
    numerics::NumericalParameters, dyn::PositionVelocity; max_duration::Float64 = 0.0, reversed_pdmp::Bool = false) where N
    #We pick a stopping threshold randomly.
    threshold = -log(rand())

    #DEPRECATED
    #We set up the initial segment.
    #Depending on the PDMP we can have many different rates.    
    #segment = evo_data.segments[vertex.segment_rate_number]

    #We update the state and forward rate and its integral until we reach termination, i.e. the max time or the threshold. A zero max_time means we do not bound time.
    update_position!(pdmp, segment, state, max_duration, evo_data, numerics, threshold, numerics.position_method, reversed_pdmp = reversed_pdmp)

    #We store the current position and time. In the backwards iteration they are altered.
    evo_data.fwd_position .= state.position
    time = segment.time

    #We update the reverse integral and reverse rate for the time.
    compute_backward_approximated_integral!(pdmp, segment, state, evo_data, numerics, numerics.position_method, reversed_pdmp = reversed_pdmp)
    
    #We restore the end position and time.
    state.position .= evo_data.fwd_position
    segment.time = time
    nothing
end
function fetch_rates!(rates::MVector, pdmp::PDMP{BPS}, state::SplitState, evolution_data::BPSEvoData, 
    numerics::NumericalParameters, dyn::PositionVelocity; reverse::Bool = false, reversed_pdmp::Bool = false)
    fetch_evo_data!(pdmp, evolution_data, numerics, state, dyn)

    k = -dot(evolution_data.gradient, state.auxiliary)

    #Finally we return the apropriate rate, depending on our pdmp and possible reversal.
    if (reversed_pdmp && reverse)||(!reversed_pdmp && !reverse)
        #We update the acutal rate.
        rates[1] = max(0.0, k)
    else
        rates[1] = max(0.0, -k)
    end
    return k
end

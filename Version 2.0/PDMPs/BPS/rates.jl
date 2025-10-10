function fetch_rates!(rates::MVector{N, Float64}, pdmp::PDMP{BPS}, state::SplitState, evolution_data::BPSEvoData, 
    numerics::NumericalParameters, dyn::PositionVelocity; reverse::Bool = false, reversed_pdmp::Bool = false, adaptive_fwd::Bool = false) where N

    fetch_evo_data!(pdmp, evolution_data, numerics, state, dyn)

    k = -dot(evolution_data.gradient, state.auxiliary)

    #Finally we return the apropriate rate, depending on our pdmp and possible reversal.
    if (reversed_pdmp && reverse)||(!reversed_pdmp && !reverse)
        #We update the acutal rate.
        rates[1] = max(0.0, k)
    else
        rates[1] = max(0.0, -k)
    end
    if adaptive_fwd
        evolution_data.fwd_rho[1] = k
    else
        evolution_data.rho[1] = k
    end
    nothing
end

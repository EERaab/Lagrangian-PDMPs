function fetch_rates!(rates::MVector, pdmp::PDMP{BPS}, state::SplitState, evolution_data::BPSEvoData, numerics::NumericalParameters, dyn::PositionVelocity; reverse::Bool = false, adaptive::Bool = false, reversed_pdmp::Bool = false)
    fetch_evo_data!(pdmp, evolution_data, numerics, state, dyn)

    k = -dot(evolution_data.gradient, state.auxiliary)

    #Finally we return the apropriate rate, depending on our pdmp and possible reversal.
    if (reversed_pdmp && reverse)||(!reversed_pdmp && !reverse)
        #We update the acutal rate.
        rates[1] = max(0, k)
        #If we use adaptive methods we will adapt not based on the rates but on the "signed" rates, which must be returned
        if adaptive
            return k
        end
        return rates
    end
    #If we use adaptive methods we will adapt not based on the rates but on the "signed" rates, which must be returned
    rates[1] = max(0, -k)
    if adaptive 
        return -k
    end
    return rates
end
function reflect!(state::SplitState, pdmp::PDMP{BPS}, evo_data::EvolutionData)
    #We compute the reflection vector
    w = evo_data.gradient

    #The reflection is now trivial to compute
    state.auxiliary .-= 2*w*dot(w,state.auxiliary)/dot(w,w)
    return state.auxiliary
end

function reflect!(state::SplitState, pdmp::PDMP{Version6_2}, evo_data::EvolutionData)
    #We compute the reflection covector
    w = evo_data.point_data.gradient

    #We raise w to a vector
    invG_w = (evo_data.spectral_data.Q)*evo_data.spectral_data.Dinv*(evo_data.spectral_data.Q')*w

    #The reflection is now trivial to compute
    state.auxiliary .-= 2*invG_w*(dot(state.auxiliary, w)/dot(w, invG_w))
    #state.auxiliary .-= 2*invG_w*((state.auxiliary'*w)/(w'*invG_w))
    return state.auxiliary
end
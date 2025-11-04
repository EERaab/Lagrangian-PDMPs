function reflect!(state::SplitState, pdmp::PDMP{Version6_2, F, D,T}, evo_data::EvolutionData) where {F,D,T}
    #We compute the reflection covector
    w = evo_data.point_data.gradient

    #We compute u^i = G^{ij}(∂_j log π)
    vec = evo_data.trash_vec
    u = evo_data.trash_vec2
    mul!(vec, evo_data.spectral_data.Q',w)
    vec .*= evo_data.spectral_data.Dinv.diag
    mul!(u, evo_data.spectral_data.Q, vec)
    
    #Old version:
    #u = (evo_data.spectral_data.Q)*evo_data.spectral_data.Dinv*(evo_data.spectral_data.Q')*w

    #The reflection is now trivial to compute
    projection_factor = 2.0*dot(state.auxiliary, w)/dot(w, u)
    state.auxiliary .-= projection_factor .* u
    return state.auxiliary
end
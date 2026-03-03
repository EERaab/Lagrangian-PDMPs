#All Lagrangian-type methods have the same (conditional) density kernel.
#This allows us to define broad methods to sample the velocity or determine the full density kernel for all such PDMPs .
#Because the soft abs metric (and its inverse) are known through its spectral decomp, and the inverse metric is the covariance
#we can adopt a simple allocation free sampling of N(0, Σ) where Σ = Q ≀ D^{-1} ≀ Q'
"""
    normal_distr!(vec::Vector{Float64}, Q::Matrix{Float64}, D::Diagonal{Float64, Vector{Float64}}, trash_vec::Vector{Float64})

Given a spectral decomposition Σ = Q*D*Q' the function samples N(0, Σ) and updates 'vec' with the outcome. 
"""
function normal_distr!(vec::Vector{Float64}, Q::Matrix{Float64}, D::Diagonal{Float64, Vector{Float64}}, trash_vec::Vector{Float64})
    @inbounds for i in eachindex(trash_vec)
        trash_vec[i] = randn()
    end
    trash_vec .*= sqrt.(D.diag)
    mul!(vec, Q, trash_vec)
    return vec
end


function sample_auxiliary_in_place!(vel::Vector{Float64}, pdmp::PDMP{M, F, D, T}, position::Array{Float64,1}, evo_data, numerics::NumericalParameters) where {M<:Lagrangian_Method, F, D, T}
    #We determine the hessian
    hessian = evo_data.core.point_data.position_update_data.value
    numerics.derivatives.hessian!(hessian, position)
    
    #We determine the spectral data associated to the hessian. 
    fetch_spectral_data!(evo_data.core.spectral_data, hessian, pdmp.method.hardness)

    Q = evo_data.core.spectral_data.Q
    Dinv = evo_data.core.spectral_data.Dinv
    return normal_distr!(vel, Q, Dinv, evo_data.workspace.vector1)
end

function sample_auxiliary!(pdmp::PDMP{M, F, D, T}, position::Array{Float64,1}, evo_data, numerics::NumericalParameters) where {M<:Lagrangian_Method, F, D, T}
    vel = zeros(Float64, pdmp.target.dimension)
    sample_auxiliary_in_place!(vel, pdmp, position, evo_data, numerics)
    return vel
end


function auxiliary_kernel!(pdmp::PDMP{M, F, D, T}, state::SplitState, evo_data, numerics::NumericalParameters)  where {M<:Lagrangian_Method, F, D, T}
    #We determine the hessian
    hessian = evo_data.core.point_data.position_update_data.value
    numerics.derivatives.hessian!(hessian, state.position)

    #We determine the spectral data associated to the hessian.
    fetch_spectral_data!(evo_data.core.spectral_data, hessian, pdmp.method.hardness)

    Dinv = evo_data.core.spectral_data.Dinv
    
    L = evo_data.workspace.vector1
    mul!(L, evo_data.core.spectral_data.Q', state.auxiliary)

    md = evo_data.workspace.vector2
    md .= L
    md ./= Dinv.diag

    return exp(-dot(L, md)/2)/sqrt(abs(det(Dinv)))
end


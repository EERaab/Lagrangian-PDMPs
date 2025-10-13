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
    trash_vec .*= D.diag
    mul!(vec, Q, trash_vec)
    return vec
end


function sample_auxiliary_in_place!(vel::Vector{Float64}, pdmp::PDMP{<:Lagrangian_Method}, position::Array{Float64,1}, evo_data::LagrangianEvoData, numerics::NumericalParameters)
    #We determine the hessian
    fetch_hessian!(evo_data.point_data, pdmp.target.log_density, position, numerics.diff_method)
    
    #We determine the spectral data associated to the hessian. It has the unfortunate name evo_data.point_data.position_update_data.value
    fetch_spectral_data!(evo_data.spectral_data, evo_data.point_data.position_update_data.value, pdmp.method.hardness)

    Q = evo_data.spectral_data.Q
    Dinv = evo_data.spectral_data.Dinv
    return normal_distr!(vel, Q, Dinv, evo_data.trash_vec)
end

function sample_auxiliary!(pdmp::PDMP{<:Lagrangian_Method}, position::Array{Float64,1}, evo_data::LagrangianEvoData, numerics::NumericalParameters)
    vel = similar(position)
    sample_auxiliary_in_place!(vel, pdmp, position, evo_data, numerics)
    return vel
end


function auxiliary_kernel!(pdmp::PDMP{<:Lagrangian_Method}, state::SplitState, evo_data::LagrangianEvoData, numerics::NumericalParameters)
    #We determine the hessian
    fetch_hessian!(evo_data.point_data, pdmp.target.log_density, state.position, numerics.diff_method)
    
    #We determine the spectral data associated to the hessian. It has the unfortunate name evo_data.point_data.position_update_data.value
    hessian = evo_data.point_data.position_update_data.value
    fetch_spectral_data!(evo_data.spectral_data, hessian, pdmp.method.hardness)

    Dinv = evo_data.spectral_data.Dinv
    L = evo_data.spectral_data.Q'*state.auxiliary

    return exp(-dot(L, inv(Dinv), L)/2)/sqrt(abs(det(Dinv)))
    #L = evo_data.spectral_data.Q'*state.auxiliary

    #return exp(-(L'*inv(evo_data.spectral_data.Dinv)*L)/2)/sqrt(abs(det(evo_data.spectral_data.Dinv)))
end


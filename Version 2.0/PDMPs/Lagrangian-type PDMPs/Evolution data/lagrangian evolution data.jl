#While the numerical parameters specify "how" we compute derivatives and etc, the specific values that we compute are stored in EvolutionData structures.
#Because of the complexity in the Lagrangian methods we shall need several "sub-structures".

#Conveniently all Lagrangian-type methods share the same basic evolution data.

    #The pointwise numerical derivatives are stored in PointData.
    mutable struct PointData
        gradient::Vector{Float64}
        position_update_data::DiffResults.MutableDiffResult
        velocity_update_data::DiffResults.MutableDiffResult
    end

    #Given the pointwise data we also store spectral data about the hessian.
    struct SpectralData
        Q::Matrix{Float64}
        jmatrix::Matrix{Float64}
        Dinv::Diagonal{Float64, Vector{Float64}}
        ws::HermitianEigenWs{Float64, Matrix{Float64}, Float64}
    end

    #Given the spectral data we construct some tensors that are used to compute the rates in the velocity update.
    #Their exact fields that we need to include will depend on the PDMP so we just define a abstract type here
    abstract type EvoTensors 
    end



#Finally we combine the above three objects into a single structure
struct LagrangianEvoData{T}<:EvolutionData{Lagrangian_Method} where T<:EvoTensors
    point_data::PointData
    spectral_data::SpectralData
    evo_tensors::T
    fwd_position::Vector{Float64} #used in backkward integral approximations
    trash_vec::Vector{Float64} #used to compute matrix products while avoiding allocations
end


#Evo-data initialization has been moved into the PDMPs.
function initialize_evolution_data(pdmp::PDMP{<:Lagrangian_Method})
    dim = pdmp.target.dimension
    ini_pd = initialize_point_data(dim)
    ini_sd = initialize_spectral_data(dim)
    ini_et = initialize_evo_tensors(pdmp)
    ini_fp = zeros(Float64, dim)
    ini_tv = zeros(Float64, dim)
    return LagrangianEvoData(ini_pd, ini_sd, ini_et, ini_fp, ini_tv)
end

#EVO tensors has been moved into the PDMPs.
#include("evo tensors.jl") 

include("point data.jl")
include("spectral data.jl")
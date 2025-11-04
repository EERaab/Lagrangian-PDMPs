
#While the numerical parameters specify "how" we compute derivatives and etc, the specific values that we compute are stored in EvolutionData structures.
#Because of the complexity in the Lagrangian methods we shall need several "sub-structures".

#Conveniently all Lagrangian-type methods share the same basic evolution data.

    #The pointwise numerical derivatives are stored in PointData.
    mutable struct PointData{DRP, DRV}
        gradient::Vector{Float64}
        position_update_data::DRP
        velocity_update_data::DRV
    end

    #Given the pointwise data we also store spectral data about the hessian.
    struct SpectralData
        Q::Matrix{Float64}
        jmatrix::Matrix{Float64} #should be symmetric
        Dinv::Diagonal{Float64, Vector{Float64}}
        ws::HermitianEigenWs{Float64, Matrix{Float64}, Float64}
    end

    #Given the spectral data we construct some tensors that are used to compute the rates in the velocity update.
    #Their exact fields that we need to include will depend on the PDMP so we just define a abstract type here

struct LagrangianEvoData{T, S, N, U, PD}<:EvolutionData
    point_data::PD
    spectral_data::SpectralData
    evo_tensors::T
    segments::S
    #Avoiding allocations:
    fwd_position::Vector{Float64} #used in backkward integral approximations
    trash_vec::Vector{Float64} #used to compute matrix products while avoiding allocations
    trash_vec2::Vector{Float64} #used to compute vector products while avoiding allocations
    trash_matrix1::Matrix{Float64}
    trash_matrix2::Matrix{Float64}
    #For use in the velocity ODE-method
    long_trash_vector::Vector{Float64}
    #adaptive methods
    adaptive_data::AdaptiveData{N}
    #velocity methods
    velocity_ode_parameters::Tuple{MVector{1, Float64}, T, Base.RefValue{Bool}, Vector{Float64}, Matrix{Float64}, Symbol}
    velocity_u0::Vector{Float64}
    integrator::U
end

include("integrator split rho.jl")
include("evo tensors.jl")

#Evo-data initialization has been moved into the PDMPs.
function initialize_evolution_data(pdmp::PDMP{M, F, D, T}, nums::NumericalParameters) where {M<:Lagrangian_Method, F, D, T}
    dim = pdmp.target.dimension
    ini_pd = initialize_point_data(dim)
    ini_sd = initialize_spectral_data(dim)
    ini_et = EvoTensors(dim)
    if M == Lagrangian
        segments = Dict{Int64, Segment{1}}(1 => Segment{1}())
    elseif M == Version6_2
        segments = Dict{Int64, Union{Segment{1}, Segment{2}}}(1 => Segment{1}(), 2 => Segment{2}())
    end
    #Avoiding allocations:
    fwd_position = zeros(Float64, dim) #used to store forward position when computing reversal in position
    trash_vec = zeros(Float64, dim) 
    trash_vec2 = zeros(Float64, dim) 
    trash_mtr1 = zeros(Float64, dim, dim)
    trash_mtr2 = zeros(Float64, dim, dim)
    #For use in the velocity ODE-method
    long_trash_vector = zeros(Float64, dim + 3)
    #adaptive methods
    st = SplitState(zeros(Float64, dim), zeros(Float64, dim), Base.RefValue{Int64}(1))
    ada = AdaptiveData{dim}(adaptive_state = st)
    #velocity methods
    (M == Lagrangian) ? method_symbol = :Lagrangian : method_symbol = :Version6_2
    velocity_ode_parameters = (MVector(0.0), ini_et, Base.RefValue(false), trash_vec, trash_mtr1, method_symbol)
    velocity_u0 = zeros(Float64, dim + 3)
    integrator = initialize_integrator(velocity_ode_parameters, nums, velocity_u0)
    return LagrangianEvoData(ini_pd, ini_sd, ini_et, segments, fwd_position, trash_vec, trash_vec2, trash_mtr1, 
        trash_mtr2, long_trash_vector, ada, velocity_ode_parameters, velocity_u0, integrator)
end

include("point data.jl")
include("spectral data.jl")
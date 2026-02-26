#struct LagrangianNumerics{P, A, D}<:NumericalParameters
#    position_method::P
#    auxiliary_method::A
#    derivatives::D
#
#    function LagrangianNumerics(pdmp::PDMP{M, F, D,T};
#        position_method = VTPiecewiseConstant(0.01), 
#        auxiliary_method = (M ≠ SplitVelocity) ? AutoTsit5(Rosenbrock23()) : Roots.Order1(), 
#        derivatives = default_derivatives(pdmp)
#        ) where {M, F, D, T}
#        new{typeof(position_method), typeof(auxiliary_method), typeof(derivatives)}(position_method, auxiliary_method, derivatives)
#    end
#end

struct VelocityODEData{T, U}
    long_vector::Vector{Float64}
    velocity_ode_parameters::Tuple{MVector{1, Float64}, T, Base.RefValue{Bool}, Vector{Float64}, Matrix{Float64}, Symbol}
    velocity_u0::Vector{Float64}
    integrator::U

    function VelocityODEData(pdmp::PDMP{M, F, D, T}, nums::NumericalParameters, t_vec, t_mtr, ini_et) where {M<:Lagrangian_Method, F, D, T}
        dim = pdmp.target.dimension
        long_vector = zeros(Float64, dim + 3)
        if M == Lagrangian
            method_symbol = :Lagrangian
        elseif M == Version6_2
            method_symbol = :Version6_2
        else
            error("Not implemented for velocity ODE.")
        end
        velocity_ode_parameters = (MVector(0.0), ini_et, Base.RefValue(false), t_vec, t_mtr, method_symbol)
        velocity_u0 = zeros(Float64, dim + 3)
        integrator = initialize_integrator(velocity_ode_parameters, nums, velocity_u0)
        new{typeof(ini_et), typeof(integrator)}(long_vector, velocity_ode_parameters, velocity_u0, integrator)
    end
end


struct LagrangianEvoData{S, T, N, U, PD}<:EvolutionData
    segments::S
    core::LagrangianCoreData{PD, T}
    workspace::LagrangianWorkspaceVariables{N}
    velocity_ode_data::VelocityODEData{T, U}

    function LagrangianEvoData(pdmp::PDMP{M, F, D, T}, nums::NumericalParameters) where {M<:Lagrangian_Method, F, D, T}
        dim = pdmp.target.dimension
        if M == Lagrangian
            segments = Dict{Int64, Segment{1}}(1 => Segment{1}())
        elseif M == Version6_2
            segments = Dict{Int64, Union{Segment{1}, Segment{2}}}(1 => Segment{1}(), 2 => Segment{2}())
        else
            error("Unimplemented!")
        end
        core = LagrangianCoreData(dim)
        workspace = LagrangianWorkspaceVariables(dim)
        vod = VelocityODEData(pdmp, nums, workspace.vector1, workspace.matrix1, core.evo_tensors)
        new{typeof(segments), typeof(core.evo_tensors), dim, typeof(vod.integrator), typeof(core).parameters[1]}(segments, core, workspace, vod)
    end
end

include("point data.jl")
include("integrator split rho.jl")
include("velocity dynamics split rho.jl")
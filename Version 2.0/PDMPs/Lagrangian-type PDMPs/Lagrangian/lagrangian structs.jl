

#The Lagrangian method requires us to specify the hardness.
@kwdef struct Lagrangian<:Lagrangian_Method
    hardness::Float64 = 0.5
end

#The numerics of the Lagrangian method is simple enough. An ODE-solver and a method for quadrature have to be specified. 
#Additionally we can use position/auxiliary types in some cases, and a variety of differentiation methods.
#For now we leave only the position and velocity methods in here.
@kwdef struct LagrangianNumerics<:NumericalParameters{Lagrangian}
    #position_type::Type = Array{Float64,1}
    #auxiliary_type::Type = Array{Float64,1}
    position_method::PositionMethod = VTPiecewiseConstant(0.01)
    auxiliary_method  = AutoTsit5(Rosenbrock23())
    diff_method::DifferentiationMethod = ForwardDer()
end

function select_dynamic(segment::PathSegmentValues{Lagrangian, N}) where N
    if segment.dyn isa PositionVelocity
        return VelocityODE{Lagrangian}()
    end
    return PositionVelocity()
end

function reverse_acceptance_rate_factor(segment::PathSegmentValues{Lagrangian, N}, old_dynamic::DynType, initial::Bool) where N
    if initial
        return 1.0
    end
    return segment.reverse_rates[1]
end

function forward_acceptance_rate_factor(segment::PathSegmentValues{Lagrangian, N}, new_dynamic::DynType, final::Bool) where N
    if final
        return 1.0
    end
    return segment.forward_rates[1]
end

function rate_number(pdmp::PDMP{Lagrangian}, dyn::DynType)::Int
    return 1
end

include("evo data.jl")
include("position evaluation.jl")
include("velocity evaluation.jl")
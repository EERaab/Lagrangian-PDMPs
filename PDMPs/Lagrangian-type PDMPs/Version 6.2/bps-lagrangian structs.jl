#For the BPS-Lagrangian method we have two rates in the position dynamic.
#The first is the BPS-event rate, the second the Lagrangian-type event rate

@kwdef struct Version6_2<:Lagrangian_Method
    hardness::Float64 = 0.5
end

#The numerics of the Lagrangian method is simple enough. An ODE-solver and a method for quadrature have to be specified. 
#Additionally we can use position/auxiliary types in some cases, and a variety of differentiation methods.
#For now we leave only the position and velocity methods in here.
@kwdef struct Version6_2Numerics<:NumericalParameters{Version6_2}
    #position_type::Type = Array{Float64,1}
    #auxiliary_type::Type = Array{Float64,1}
    position_method::PositionMethod = VTPiecewiseConstant(0.01)
    auxiliary_method  = AutoTsit5(Rosenbrock23())
    diff_method::DifferentiationMethod = ForwardDer()
end

function select_dynamic(segment::PathSegmentValues{Version6_2, N}) where N
    if segment.dyn isa PositionVelocity
        rel_prob = segment.forward_rates[1]/sum(segment.forward_rates)
        if rel_prob < Inf
            if rand() < rel_prob
                return GradientReflection{Version6_2}()
            else
                return VelocityODE{Version6_2}()
            end
        else
            error("Inexact rates (both zero!)")
        end
    end
    return PositionVelocity()
end

function reverse_acceptance_rate_factor(segment::PathSegmentValues{Version6_2, N}, old_dynamic::DynType, initial::Bool) where N
    if initial || (segment.dyn isa GradientReflection{Version6_2})
        return 1.0
    end
    if segment.dyn isa PositionVelocity
        if old_dynamic isa GradientReflection{Version6_2}
            return segment.reverse_rates[1]
        elseif old_dynamic isa VelocityODE{Version6_2}
            return segment.reverse_rates[2]
        end
    elseif (segment.dyn isa VelocityODE{Version6_2}) && (old_dynamic isa PositionVelocity)
        return segment.reverse_rates[1]
    end
    error("Incorrect dynamic selected. Old: "*string(old_dynamic)*" and current: "*string(segment.dyn))
end

function forward_acceptance_rate_factor(segment::PathSegmentValues{Version6_2, N}, new_dynamic::DynType, final::Bool) where N
    if final || (segment.dyn isa GradientReflection{Version6_2})
        return 1.0
    end
    if segment.dyn isa PositionVelocity
        if new_dynamic isa GradientReflection{Version6_2}
            return segment.forward_rates[1]
        elseif new_dynamic isa VelocityODE{Version6_2}
            return segment.forward_rates[2]
        end
    elseif (segment.dyn isa VelocityODE{Version6_2}) && (new_dynamic isa PositionVelocity)
        return segment.forward_rates[1]
    end
    error("Incorrect dynamic selected. New: "*string(new_dynamic)*" and current: "*string(segment.dyn))
end

function rate_number(pdmp::PDMP{Version6_2}, dyn::DynType)::Int
    if dyn isa PositionVelocity
        return 2
    end
    return 1
end

include("evo data.jl")
include("position evaluation.jl")
include("velocity evaluation.jl")
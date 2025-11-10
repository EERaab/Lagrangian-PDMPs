abstract type PositionMethod
end

abstract type VT<:PositionMethod
end

@kwdef struct VTPiecewiseConstant<:VT
    step_size::Float64 = 0.01
end

@kwdef struct VTAdaptivePiecewiseConstant<:VT
    tolerance::Float64 = 0.01
    step_guess::Float64 = 0.01
    max_adaptive_rel_size::Float64 = 10.0
end

include("piecewise constant approximation.jl")
include("adaptive piecewise constant.jl")
#We seek to study probability densities - these are our targets.
#We take their logarithm and dimension as our basic object to work with.
struct TargetData{F} 
    log_density::F
    dimension::Integer
end

#A PDMP is a PDMP method and a target on which that method is applied. Together the two determine the PDMP fully.
#Abstractly they tell us what the method does.
abstract type PDMP_Method
end

#On occasion we shall want to reverse the PDMP, and this Z2 transf. can be included in the original PDMP.
@kwdef struct PDMP{T<:PDMP_Method, F}
    method::T
    target::TargetData{F}
    reversed::Bool = false
end

#PDMPs are defined by evolving according to some dynamic along a segment (or instantaneously at events). 
#There are many possible dynamics, but each path segment (and every event) follows one or another.
abstract type DynType
end

#Depending on the dynamic we have different evaluations.
@kwdef mutable struct PathSegmentValues{N, T<:DynType}
    dyn::T
    time::Float64 = 0.0
    forward_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
    reverse_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
    forward_rate_integral::Float64 = 0.0
    reverse_rate_integral::Float64 = 0.0
end

#We shall want to reuse our PathSegmentValues for repeated evolutions according to the same dynamics
#Therefore we build a DiGraph to encode the possible transitions and states.
struct MethodDiGraph{F}
    vertices::Vector{PathSegmentValues}
    edges::Vector{F}
end





#Each PDMP has a corresponding "reversed" PDMP. Sometimes it is useful to construct the reverse.
function reverse(pdmp::PDMP)
    return PDMP(pdmp.method, pdmp.target, !pdmp.reversed)
end

#Each PDMP method has its own numerical data structures related to how rates etc are computed.
#These effectively define "how" we do computations.
abstract type NumericalParameters{T<:PDMP_Method}
end

#Each PDMP also will compute some data that will be updated as we move along.
abstract type EvolutionData{T<:PDMP_Method}
end

#For all PDMPS we consider we shall use binary states where we have a position and an auxiliary that is the velocity.
#In principle the auxiliary could be the momentum or some such, hence the unfortunate naming convention.
struct SplitState
    position::Array{Float64,1}
    auxiliary::Array{Float64,1}
end

#It is useful to be able to initialize a binary state for a PDMP.
function initialize_binary_state!(pdmp::PDMP{T}, evo_data::EvolutionData, numerics::NumericalParameters; initial_position::Vector{Float64} = rand(pdmp.target.dimension)) where T<:PDMP_Method
    return SplitState(initial_position, sample_auxiliary!(pdmp, initial_position, evo_data, numerics))
end

#All PDMP methods will need to compute derivatives. How this is done can be define by some Differentiation Method
abstract type DifferentiationMethod
end

#ForwardDer uses ForwardDiff
struct ForwardDer<:DifferentiationMethod
end

#AnalyticalDer assumes the user provides the relevant derivative functions.
#These are assumed to be in-place!
struct AnalyticalDer{F1,F2,F3,F4}<:DifferentiationMethod
    gradient!::F1
    hessian!::F2
    third_order_directional!::F3
    third_order_full!::F4
end

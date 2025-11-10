#All but one PDMPs we consider here is a Lagrangian-type pdmp, and we include this as an abstract type.
#All Lagrangian methods share the same set of evolution data, though the data is treated differently
abstract type Lagrangian_Method<:PDMP_Method
end

include("auxiliary kernel.jl")
include("Evolution data/core evo data.jl")

#In all Lagrangian methods we need four types of derivatives.
struct LagrangianDerivatives{F1, F2, F3, F4}
    gradient!::F1
    hessian!::F2
    full_third_order!::F3
    directional_third_order!::F4
    #We have built our framework to utilize the simultaneous computation of low-order and higher order derivatives.
    #For various reasons this is only used for the hessian/third order. 
    #I.e. the gradient is only assumed to be computed when explicitly called.
    #Not all methods derivatives are built in this ways so we track this
    hessian_computed_in_higher_order::Bool
end

#We use ForwardDiff to generate default derivatives so that a user doesn't *need* to give some form of derivative.
include("default derivatives.jl")
  
#The specifics of each method is contained in separate structures.
include("Lagrangian/lagrangian structs.jl")
include("Version 6.2/bps-lagrangian structs.jl")

#Lagrangian and Version 6.2 both use a shared numerics structure.
include("Velocity ODE data/Lagrangian numerics.jl")  

function initialize_evolution_data(pdmp::PDMP{M, F, D, T}, nums::NumericalParameters) where {M<:Lagrangian_Method, F, D, T}
    if M == Lagrangian || M == Version6_2
        return LagrangianEvoData(pdmp, nums)
    elseif M == :PLACEHOLDER
        error("Unimplemented")
    else
        error("Unimplemented")
    end
end

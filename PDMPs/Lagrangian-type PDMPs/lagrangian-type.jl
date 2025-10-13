#All but one PDMPs we consider here is a Lagrangian-type pdmp, and we include this as an abstract type.
#All Lagrangian methods share the same set of evolution data, though the data is treated differently
abstract type Lagrangian_Method<:PDMP_Method
end

include("Evolution data/lagrangian evolution data.jl")
include("auxiliary kernel.jl")

#include("BPS-Lagrangian/bps-lagrangian structs.jl")
include("Lagrangian/lagrangian structs.jl")
#include("Version 6.2/bps-lagrangian structs.jl")
#include("Lagrangian Gradient Reflection/lagrangian gradient reflection structs.jl")

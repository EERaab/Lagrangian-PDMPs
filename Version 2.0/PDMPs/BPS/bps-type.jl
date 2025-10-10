#This file is really unnecessary but in order to make the overall structure more coherent we mimic the structure of the Lagrangian types.

abstract type BPS_Method<:PDMP_Method
end

include("bps structs.jl")
include("bps evo data.jl")
include("auxiliary kernel.jl")
include("rates.jl")
include("velocity evaluation.jl")
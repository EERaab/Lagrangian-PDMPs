#For the BPS-Lagrangian method we have two rates in the position dynamic.
#The first is the BPS-event rate, the second the Lagrangian-type event rate

@kwdef struct Version6_2<:Lagrangian_Method
    hardness::Float64 = 0.5
end

function generate_pdmp_graph(method::Version6_2, target::TargetData{F})::PDMP_DiGraph where F
    vertices = [MethodVertex(1, 2), MethodVertex(2, 1)]
    edges = [[MethodEdge(1,1,1), MethodEdge(1,2,2)], [MethodEdge(2,1,2)]]
    return PDMP_DiGraph(vertices, edges, (PositionVelocity(), VelocityODE()), (GradientReflection(), Identity()))
end

include("position evaluation.jl")
include("velocity evaluation.jl")
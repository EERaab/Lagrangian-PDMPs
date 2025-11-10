#The Lagrangian method requires us to specify the hardness.
@kwdef struct Lagrangian<:Lagrangian_Method
    hardness::Float64 = 0.5
end

function generate_pdmp_graph(method::Lagrangian, target::TargetData{F})::PDMP_DiGraph where F
    vertices = [MethodVertex(1,1), MethodVertex(2, 1)]
    edges = [[MethodEdge(1,2,1)],[MethodEdge(2,1,1)]]
    return PDMP_DiGraph(vertices, edges, (PositionVelocity(), VelocityODE()), (Identity(),))
end

include("position evaluation.jl")
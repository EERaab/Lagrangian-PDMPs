#The other subtype, apart from the Lagrangian type is just BPS, with a single pdmp type. 
struct BPS<:BPS_Method
end

@kwdef struct BPSNumerics<:NumericalParameters
    #position_type::Type = Vector{Float64}
    #auxiliary_type::Type = Vector{Float64}
    position_method::PositionMethod = VTPiecewiseConstant(0.01)
    diff_method::DifferentiationMethod = ForwardDer()
end

function generate_pdmp_graph(method::BPS, target::TargetData{F})::PDMP_DiGraph where F
    vertices = [MethodVertex(PositionVelocity(),1)]
    edges = Vector{MethodEdge{GradientReflection}}[[MethodEdge(1,1, GradientReflection())]]
    return PDMP_DiGraph(vertices, edges)
end
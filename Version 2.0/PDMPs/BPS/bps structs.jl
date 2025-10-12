#The other subtype, apart from the Lagrangian type is just BPS, with a single pdmp type. 
struct BPS<:BPS_Method
end

@kwdef struct BPSNumerics{P<:PositionMethod,D<:DifferentiationMethod}<:NumericalParameters
    #position_type::Type = Vector{Float64}
    #auxiliary_type::Type = Vector{Float64}
    position_method::P = VTPiecewiseConstant(0.01)
    diff_method::D = ForwardDer()
end

function generate_pdmp_graph(method::BPS, target::TargetData{F})::PDMP_DiGraph where F
    vertices = [MethodVertex(1,1)]
    edges = [[MethodEdge(1,1,1)]]
    return PDMP_DiGraph(vertices, edges, (PositionVelocity(),), (GradientReflection(),))
end
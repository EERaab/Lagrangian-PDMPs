#The other subtype, apart from the Lagrangian type is just BPS, with a single pdmp type. 
struct BPS<:BPS_Method
end

struct BPSNumerics{P<:PositionMethod, D}<:NumericalParameters
    position_method::P
    gradient!::D

    function BPSNumerics(pdmp::PDMP{M, F, X, T}; position_method = VTPiecewiseConstant(0.01), gradient! = default_grad_bps(pdmp)) where {M<:BPS_Method, F, X, T}
        new{typeof(position_method), typeof(gradient!)}(position_method, gradient!)
    end
end

function default_grad_bps(pdmp::PDMP{M, F, D, T}) where {M<:BPS_Method, F, D, T}
    return (grad, X) -> ForwardDiff.gradient!(grad, pdmp.target.log_density, X)
end

function generate_pdmp_graph(method::BPS, target::TargetData{F})::PDMP_DiGraph where F
    vertices = [MethodVertex(1,1)]
    edges = [[MethodEdge(1,1,1)]]
    return PDMP_DiGraph(vertices, edges, (PositionVelocity(),), (GradientReflection(),))
end
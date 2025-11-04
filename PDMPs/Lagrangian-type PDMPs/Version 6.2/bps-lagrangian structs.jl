#For the BPS-Lagrangian method we have two rates in the position dynamic.
#The first is the BPS-event rate, the second the Lagrangian-type event rate

@kwdef struct Version6_2<:Lagrangian_Method
    hardness::Float64 = 0.5
end

#The numerics of the Lagrangian method is simple enough. An ODE-solver and a method for quadrature have to be specified. 
#Additionally we can use position/auxiliary types in some cases, and a variety of differentiation methods.
#For now we leave only the position and velocity methods in here.
@kwdef struct Version6_2Numerics{P<:PositionMethod, A, D<:DifferentiationMethod}<:NumericalParameters
    #position_type::Type = Array{Float64,1}
    #auxiliary_type::Type = Array{Float64,1}
    position_method::P = VTPiecewiseConstant(0.01)
    auxiliary_method::A  = AutoTsit5(Rosenbrock23())
    diff_method::D = ForwardDer()
end

function generate_pdmp_graph(method::Version6_2, target::TargetData{F})::PDMP_DiGraph where F
    vertices = [MethodVertex(1, 2), MethodVertex(2, 1)]
    edges = [[MethodEdge(1,1,1), MethodEdge(1,2,2)], [MethodEdge(2,1,2)]]
    return PDMP_DiGraph(vertices, edges, (PositionVelocity(), VelocityODE()), (GradientReflection(), Identity()))
end

include("evo data.jl")
include("position evaluation.jl")
include("velocity evaluation.jl")
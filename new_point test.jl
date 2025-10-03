using StaticArrays
#PDMPs are defined by evolving according to some dynamic along a segment (or instantaneously at events). 
#There are many possible dynamics, but each path segment (and every event) follows one or another.
abstract type DynType
end
struct PositionVelocity <:DynType
end
struct VelocityODE <:DynType
end
mutable struct IntegratedPartialVelocity <:DynType
    index::Int
end

abstract type TransitionType
end
struct GradientReflection<:TransitionType
end
struct Identity<:TransitionType
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
    edges::Vector{MethodEdge{F}}
end

struct MethodEdge{F}
    target_vertex::Integer
    transition_map::F
end

#Version 6.2 (CA-BPS)
    #Vertices correspond to non-instant evolution (this is a computational notion - mathematically we can view all nearly evolutions as instantaneous)
    position_evo_v62 = PathSegmentValues{2}(dyn = PositionVelocity())
    velocity_evo_v62 = PathSegmentValues{1}(dyn = VelocityODE())
    vertices = [position_evo_v62, velocity_evo_v62]

    #We store the edges in the following way
    #edges[1] = edges that start in 1 = an array of edges, each with an associated transition
    #edges[2] = edges that start in 2 = an array of edges, each with an associated transition
    #The order of the edges in edges[i] is such that it corresponds to the order in the fwd rates of the PathSegmentValues of vertex i.
    #That is to say edges[i][j] = is the event that we transition from i through the j:th transition.
    #The resulting vertex is thus not obvious 
    xx_edge = MethodEdge(1, GradientReflection()) #the xx edge is a gradient reflection from position evo to position evo
    xv_edge = (2, Identity())
    vx_edge = (1, Identity())
    CA_BPS_Graph = MethodDiGraph(vertices, edges)


whatever
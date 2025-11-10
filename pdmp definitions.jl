using StaticArrays
abstract type DynType
end

#We define some dynamics
    struct PositionVelocity<:DynType
    end

    struct VelocityODE<:DynType
    end

    struct VelocityPartial <:DynType
        component::Int64
    end


#The structures above correspond to our different (analytical) flows. These are then handled differently depending on our choice of numerics.


abstract type TransitionType
end
#We define some transition types: These encode the position, auxiliary transformations of the state
#The transformation associated to the split index (i.e. the α → β) are encoded separately.
#In theory these could contain some information, but in present implementations they could be replaced by strings, integers or whatever

    struct GradientReflection<:TransitionType
    end

    struct Identity<:TransitionType
    end

#We define enlarged DiGraph structures
#The vertices correspond to the α of the paper. As such each should contain a 'Segment' (since for each α we get a type of flow to evaluate), 
#but there are some isomorphisms between flows, so we get a 'Segment' for each equivalence class ̄α. Therefore we have the vertices contain ̄α 
#so as to indicate which class they belong to.

    #The edges are our transitions between αs (which can act trivially α → α, as in BPS).
    #They contain a transition function that tells us what the state transforms into.
    #Edges pointing from the i:th vertex are in mth_dg.edges[i], 
    #Here mth_dg.edges[i][j] corresponds to the j:th transition of the i:th vertex, 

    struct MethodVertex
        dynamic_number::Int64
        segment_rate_number::Int64
    end

    #We take the edges to carry both the target and base vertex, i.e. each edge is a e = (α → β)
    #and so we expect mth_dg.edges[i] = {e ∈ edge_set | e = i → β for some β}.

    struct MethodEdge
        base_vertex_number::Int64
        target_vertex_number::Int64
        transition_number::Int64
    end
    

struct PDMP_DiGraph{D<:Tuple,T<:Tuple}
    vertices::Vector{MethodVertex}
    edges::Vector{Vector{MethodEdge}}
    dynamics::D
    transitions::T
end

@kwdef mutable struct Segment{N}
    time::Float64 = 0.0
    forward_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
    reverse_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
    forward_rate_integral::Float64 = 0.0
    reverse_rate_integral::Float64 = 0.0
end

function reset_segment!(seg::Segment{N})::Segment{N} where N
    seg.time = 0.0
    @inbounds for i in eachindex(seg.forward_rates::MVector{N, Float64})
        seg.forward_rates[i] = 0.0
        seg.reverse_rates[i] = 0.0
    end
    seg.forward_rate_integral = 0.0
    seg.reverse_rate_integral = 0.0
    return seg
end

#We define targets:
struct TargetData{F} 
    log_density::F
    dimension::Int64
end

#Together with a PDMP method the target defines a PDMP:
abstract type PDMP_Method
end

#We can now define PDMPs.
#We shall let the PDMP and its graph also represent the reversed PDMP, since we assume the two are closely related.
#For example we know, that there exists a map 'reverse' on the edge set of the graph
# that associates to each edge E another edge E' such that EE'(x) = x and E'E(y) = y. 
@kwdef struct PDMP{M<:PDMP_Method, F, D<:Tuple,T<:Tuple}
    method::M
    target::TargetData{F}
    graph::PDMP_DiGraph{D,T} = generate_pdmp_graph(method, target)
end

#Again, by assumption the reversed PDMP has the same transitions we can get the reverse edges.
#Explicitly if e = α → β through transition T, then e_rev = β → α through T. 
#Rather than returning e_rev (which would be necessary if e_rev had a different transition from e)
#we return k s.t. graph.edges[e.target_vertex_number][k] = e_rev
function reversed_edge_number(graph::PDMP_DiGraph, edge::MethodEdge)::Int64
    n = edge.target_vertex_number
    k = 1
    for alt_edge in graph.edges[n]
        if alt_edge.target_vertex_number !== edge.base_vertex_number 
            k+=1
            continue
        end
        if alt_edge.transition_number !== edge.transition_number
            k+=1
            continue
        end
        return k #so that graph.edges[n][k] = alt_edge
    end
    #Unless we've made a mistake there will be some edge that is the reversed edge.
    error("No reversed edge could be found.")
end


#The PDMPs will have states that transform as we go along.
struct SplitState
    position::Vector{Float64} #x
    auxiliary::Vector{Float64} #v (or p)
    split_index::Base.RefValue{Int64} #α - given as a ref so we can mutate the index value
end

#In a given PDMP there are going to be some numerics dictating how approximations are made
abstract type NumericalParameters
end

#Given the numerics and the PDMP some specific set of data is used repeatedly in computations and
#should be used for in-place computations.
abstract type EvolutionData
end

#In adaptive step-size methods we use a set of temporary states to adjust step size
@kwdef struct AdaptiveData{N}
    adaptive_state::SplitState
    adaptive_fwd_rates::MVector{N, Float64} = @MVector zeros(Float64, N)
    rho::MVector{N, Float64} = @MVector zeros(Float64, N)
    fwd_rho::MVector{N, Float64} = @MVector zeros(Float64, N)
end

function initialize_state!(pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters; 
    initial_position = rand(pdmp.target.dimension), 
    initial_auxiliary = sample_auxiliary!(pdmp, initial_position, evo_data, nums),
    initial_split_index = Base.RefValue{Int64}(1))
    
    return SplitState(initial_position, initial_auxiliary, initial_split_index)
end

function copy_state(st::SplitState)
    return SplitState(copy.(st.position), copy.(st.auxiliary), Base.RefValue(st.split_index.x))
end

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
#Concretely, the segment associated to the i:th vertex is mth_dg.segments[mth_dg.vertex[i].segment_number]

    #The edges are our transitions between αs (which can act trivially α → α, as in BPS).
    #They contain a transition function that tells us what the state transforms into.
    #Edges pointing from the i:th vertex are in mth_dg.edges[i], 
    #Here mth_dg.edges[i][j] corresponds to the j:th transition of the i:th vertex, 
    #which has a rate given by mth_dg.segments[mth_dg.vertex[i].segment_number].forward_rates

    struct MethodVertex{D<:DynType}
        dynamic::D
        segment_number::Int64
    end

    struct MethodEdge{T<:TransitionType}
        target_vertex_number::Int64
        transition::T
    end
    
    @kwdef struct Segment{N}
        time::Float64 = 0.0
        forward_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
        reverse_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
        forward_rate_integral::Float64 = 0.0
        reverse_rate_integral::Float64 = 0.0
    end

struct MethodDiGraph
    vertices::Vector{MethodVertex}
    edges::Vector{Vector{MethodEdge}}
    segments::Vector{Segment}
end

#We define targets:
struct TargetData{F} 
    log_density::F
    dimension::Integer
end

#Together with a PDMP method the target defines a PDMP:
abstract type PDMP_Method
end

#We can now define PDMPs:
@kwdef struct PDMP{T<:PDMP_Method}
    method::T
    target::TargetData
    graph::MethodDiGraph = generate_method_graph(method, target)
    reversed::Bool = false
end

#The reversal of a PDMP is useful to construct:
function reverse(pdmp::PDMP)
    return PDMP(pdmp.method, pdpmp.target, pdmp.graph, !pdmp.reversed)
end

#The PDMPs will have states that transform as we go along.
struct SplitState
    position::Vector{Float64} #x
    auxiliary::Vector{Float64} #v (or p)
    split_index::Ref{Int64} #α - given as a ref so we can mutate the index value
end

#In a given PDMP there are going to be some numerics dictating how approximations are made
abstract type NumericalParameters
end

#Given the numerics and the PDMP some specific set of data is used repeatedly in computations and
#should be used for in-place computations.
abstract type EvolutionData
end


function initialize_state!(pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters; 
    initial_pos = rand(pdmp.target.dimension), 
    initial_aux = sample_auxiliary!(pdmp, initial_pos, evo_data, nums),
    initial_splindex = 1)
    
    return SplitState(initial_pos, initial_aux, initial_splindex)
end

struct BPSWorkspace{N}
    #The fields below are 'trash' Used for avoiding allocs 
    fwd_position::Vector{Float64} #used in backkward integral approximations
    #adaptive methods
    adaptive_data::AdaptiveData{N}
end

struct BPSEvoData{S, N}<:EvolutionData 
    segments::S
    workspace::BPSWorkspace{N}
    gradient::Vector{Float64}
end

function initialize_evolution_data(pdmp::PDMP{BPS, F, D, T}, nums::NumericalParameters) where {F, D, T}
    grad = zeros(Float64, pdmp.target.dimension)
    seg_dict = Dict{Int64, Segment{1}}(1=>Segment{1}())
    fwd_position = zeros(Float64, pdmp.target.dimension)
    st = SplitState(zeros(Float64, pdmp.target.dimension), zeros(Float64, pdmp.target.dimension),Base.RefValue{Int64}(1))
    ada = AdaptiveData{pdmp.target.dimension}(adaptive_state = st)
    work = BPSWorkspace(fwd_position, ada)
    return BPSEvoData(seg_dict, work, grad)
end

function fetch_evo_data!(pdmp::PDMP{BPS, F, D, T}, evo_data::BPSEvoData, numerics::NumericalParameters, state::SplitState, dyn_type::Union{DynType,TransitionType}) where {F, D, T}
    numerics.gradient!(evo_data.gradient, state.position)
    nothing
end

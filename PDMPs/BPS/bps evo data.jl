struct BPSEvoData{S, N}<:EvolutionData 
    gradient::Vector{Float64}
    segments::Dict{Int64, S}
    #The fields below are 'trash' Used for avoiding allocs 
    fwd_position::Vector{Float64} #used in backkward integral approximations
    #adaptive methods
    adaptive_data::AdaptiveData{N}
end

function initialize_evolution_data(pdmp::PDMP{<:BPS_Method}, nums::NumericalParameters)
    grad = zeros(Float64, pdmp.target.dimension)
    seg_dict = Dict{Int64, Segment{1}}(1=>Segment{1}())
    fwd_position = zeros(Float64, pdmp.target.dimension)
    st = SplitState(zeros(Float64, pdmp.target.dimension), zeros(Float64, pdmp.target.dimension),Base.RefValue{Int64}(1))
    ada = AdaptiveData{pdmp.target.dimension}(adaptive_state = st)
    return BPSEvoData(grad, seg_dict, fwd_position, ada)
end

function fetch_evo_data!(pdmp::PDMP{<:BPS_Method}, evo_data::BPSEvoData, numerics::NumericalParameters, state::SplitState, dyn_type::Union{DynType,TransitionType})
    if numerics.diff_method isa ForwardDer
        ForwardDiff.gradient!(evo_data.gradient, pdmp.target.log_density, state.position)
    elseif numerics.diff_method isa AnalyticalDer
        numerics.diff_method.gradient!(evo_data.gradient, state.position)
    else
        error("Unspecified derivative method used.")
    end
    nothing
end

struct BPSEvoData<:EvolutionData
    gradient::Vector{Float64}
    fwd_position::Vector{Float64} #used in backkward integral approximations
    adaptive_state::SplitState
end

function initialize_evolution_data(pdmp::PDMP{<:BPS_Method})
    return BPSEvoData(zeros(Float64, pdmp.target.dimension), zeros(Float64, pdmp.target.dimension), 
    SplitState(zeros(Float64, pdmp.target.dimension),zeros(Float64, pdmp.target.dimension),1))
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

function transition!(state::SplitState, pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters, edge::MethodEdge, transition::GradientReflection)
    #We get the relevant data for a gradient reflection:
    fetch_core_data!(pdmp, evo_data, nums, state, transition)

    #We introduce a shorthand definition. 
    reflect!(state, pdmp, evo_data)

    state.split_index.x = edge.target_vertex_number
    nothing
end

function transition!(state::SplitState, pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters, edge::MethodEdge, transition::Identity)
    state.split_index.x = edge.target_vertex_number
    nothing
end


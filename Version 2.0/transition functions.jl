function transition!(state::SplitState, pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters, edge::MethodEdge{GradientReflection})
    #We get the relevant data for a gradient reflection:
    fetch_evo_data!(pdmp, evo_data, nums, state, edge.transition)

    #We introduce a shorthand definition. 
    reflect!(state, pdmp, evo_data)

    state.split_index.x = edge.target_vertex_number
    nothing
end

function transition!(state::SplitState, pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters, edge::MethodEdge{Identity})
    state.split_index.x = edge.target_vertex_number
    nothing
end


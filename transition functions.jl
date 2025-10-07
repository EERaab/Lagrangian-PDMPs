function transition!(state::SplitState, pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters, transition::GradientReflection)
    #We get the relevant data for a gradient reflection:
    fetch_evo_data!(pdmp, evo_data, nums, state, edge, transition)

    #We introduce a shorthand definition. 
    reflect!(state, pdmp, evo_data)

    nothing
end

function transition!(state::SplitState, pdmp::PDMP, evo_data::EvolutionData, nums::NumericalParameters, transition::Identity)
    #Notably there is no transformation under the identity transformation.
    nothing
end
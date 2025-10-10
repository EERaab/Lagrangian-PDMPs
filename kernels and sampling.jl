#This file will soon be removed as optimizations has moved its functionality into subfolders.


#function sample_auxiliary!(pdmp::PDMP, position::Array{Float64,1}, evolution_data::EvolutionData, numerics::NumericalParameters)
    #We determine the covariance matrix for the aux distribution of the given PDMP
#    fetch_velocity_covariance_matrix!(pdmp, position, evolution_data, numerics)

    #The conditional aux distribution is v|x ∼ N(0, cov)
#    return rand(MvNormal(evolution_data.aux_covariance))
#end
#MOVED TO auxiliary kernel of BPS resp. Lagrangian Methods for specialized functionality.

function full_density_kernel!(pdmp::PDMP, state::SplitState, evolution_data::EvolutionData, numerics::NumericalParameters)
    return exp(pdmp.target.log_density(state.position))*auxiliary_kernel!(pdmp, state, evolution_data, numerics)
end

#function fetch_velocity_covariance_matrix!(pdmp::PDMP{<:BPS_Method}, position::Vector{Float64}, evo_data::BPSEvoData, numerics::BPSNumerics)
#    return evo_data.aux_covariance
#end

function sample_auxiliary!(pdmp::PDMP{<:BPS_Method}, position::Array{Float64,1}, evo_data::BPSEvoData, numerics::BPSNumerics)
    return rand(pdmp.target.dimension)
end

function auxiliary_kernel!(pdmp::PDMP{<:BPS_Method}, state::SplitState, evo_data::BPSEvoData, numerics::BPSNumerics)
    #Some constant factor is missing here. It will not matter.
    return exp(-dot(state.auxiliary, state.auxiliary)/2)
end
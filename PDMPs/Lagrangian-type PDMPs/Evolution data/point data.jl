function fetch_point_data!(point_data::PointData, pdmp::PDMP{<:Lagrangian_Method}, state::SplitState, diff_method::ForwardDer, dyn_type::PositionVelocity)
    #We introduce a shorthand
    log_density = pdmp.target.log_density

    #We define the directional hessian function t ↦ Hf(x+tv)
    hf = t -> ForwardDiff.hessian(log_density, (state.position .+ (t.*state.auxiliary)))

    #We take a directional derivative on the hessian function and store the value (hessian) and the derivative.
    point_data.position_update_data = ForwardDiff.derivative!(point_data.position_update_data, hf, 0.0)

    #We compute the gradient.
    ForwardDiff.gradient!(point_data.gradient, log_density, state.position)
    nothing
end

function fetch_point_data!(point_data::PointData, pdmp::PDMP{<:Lagrangian_Method}, state::SplitState, diff_method::AnalyticalDer, dyn_type::PositionVelocity)
    diff_method.gradient!(point_data.gradient, state.position)

    diff_method.hessian!(point_data.position_update_data.value, state.position)
    
    diff_method.third_order_directional!(point_data.position_update_data.derivs[1], state.position, state.auxiliary)
    
    nothing
end

function fetch_hessian!(point_data::PointData, log_density::Function, position::Array{Float64,1}, diff_method::ForwardDer)
    point_data.position_update_data.value = ForwardDiff.hessian!(point_data.position_update_data.value, log_density, position)
    nothing
end

function fetch_hessian!(point_data::PointData, log_density::Function, position::Array{Float64,1}, diff_method::AnalyticalDer)
    diff_method.hessian!(point_data.position_update_data.value, position)
    nothing
end

function initialize_point_data(dim::Integer)::PointData
    return PointData(zeros(Float64, dim), DiffResults.DiffResult(zeros(Float64, dim, dim),zeros(Float64, dim, dim)), DiffResults.DiffResult(zeros(Float64, dim, dim),zeros(Float64, dim, dim, dim)))
end

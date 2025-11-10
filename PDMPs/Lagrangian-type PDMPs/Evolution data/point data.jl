function fetch_point_data!(point_data::PointData, pdmp::PDMP{M, F, D, T}, state::SplitState, derivatives, dyn_type::PositionVelocity) where {M<:Lagrangian_Method, F, D, T}
    #We take a directional derivative on the hessian function and store the value (hessian) and the derivative.
    if derivatives.hessian_computed_in_higher_order
        derivatives.directional_third_order!(point_data.position_update_data, state.position, state.auxiliary)
    else
        derivatives.directional_third_order!(point_data.position_update_data.derivs[1], state.position, state.auxiliary)
        derivatives.hessian!(point_data.position_update_data.value, state.position)
    end

    #We compute the gradient.
    derivatives.gradient!(point_data.gradient, state.position)
    nothing
end

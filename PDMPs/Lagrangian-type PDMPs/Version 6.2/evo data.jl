#For ForwardDiff we use this function to fetch point data.
function fetch_point_data!(point_data::PointData, pdmp::PDMP{Version6_2, F, D, T}, state::SplitState, diff_method::ForwardDer, dyn_type::VelocityODE) where {F,D,T}
    #We introduce a shorthand
    log_density = pdmp.target.log_density
    
    #We define the hessian function x ↦ Hf(x)
    hf = u -> ForwardDiff.hessian(log_density, u)
    
    #We take a derivative on the hessian function and store the value (hessian) and the derivative.
    point_data.velocity_update_data = ForwardDiff.jacobian!(point_data.velocity_update_data, hf, state.position)

    #We compute the gradient.
    ForwardDiff.gradient!(point_data.gradient, log_density, state.position)

    nothing
end

#For analytically given derivatives we use this function to fetch point data.
function fetch_point_data!(point_data::PointData, pdmp::PDMP{Version6_2, F, D, T}, state::SplitState, diff_method::AnalyticalDer, dyn_type::VelocityODE) where {F, D, T}
    diff_method.gradient!(point_data.gradient, state.position)

    diff_method.hessian!(point_data.velocity_update_data.value, state.position)

    diff_method.third_order_full!(point_data.velocity_update_data.derivs[1], state.position)

    nothing
end


function fetch_evo_data!(pdmp::PDMP{Version6_2, F, D, T}, evo_data::LagrangianEvoData, numerics::NumericalParameters, state::SplitState, dyn::VelocityODE) where {F, D, T}
    #We determine the values of the Hessian, its Jacobian, and other relevant data at the point X.
    fetch_point_data!(evo_data.point_data, pdmp, state, numerics.diff_method, dyn)
    
    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = evo_data.point_data.velocity_update_data.value
    fetch_spectral_data!(evo_data.spectral_data, hessian, pdmp.method.hardness)

    fetch_evo_tensors!(evo_data, pdmp.target.dimension)
    nothing
end

function fetch_gradient!(point_data::PointData, pdmp::PDMP{Version6_2, F, D, T}, state::SplitState, diff_method::ForwardDer) where {F,D,T}
    #We compute the gradient.
    ForwardDiff.gradient!(point_data.gradient, pdmp.target.log_density, state.position)

    nothing
end

function fetch_gradient!(point_data::PointData, pdmp::PDMP{Version6_2, F, D, T}, state::SplitState, diff_method::AnalyticalDer) where {F,D,T}
    #We compute the gradient.
    diff_method.gradient!(point_data.gradient, state.position)
    nothing
end

function fetch_evo_data!(pdmp::PDMP{Version6_2, F, D, T}, evo_data::LagrangianEvoData, numerics::NumericalParameters, state::SplitState, transition::GradientReflection) where {F, D, T}
    #We determine the values of the Hessian and the gradient (stored in *position* data here)
    fetch_hessian!(evo_data.point_data, pdmp.target.log_density, state.position, numerics.diff_method)

    fetch_gradient!(evo_data.point_data, pdmp, state, numerics.diff_method)
    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = evo_data.point_data.position_update_data.value
    fetch_spectral_data!(evo_data.spectral_data, hessian, pdmp.method.hardness)

    nothing
end

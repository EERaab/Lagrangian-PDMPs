#For ForwardDiff we use this function to fetch point data.
function fetch_point_data!(point_data::PointData, pdmp::PDMP, state::SplitState, derivatives::LagrangianDerivatives, dyn::VelocityODE)

    if derivatives.hessian_computed_in_higher_order
        derivatives.full_third_order!(point_data.velocity_update_data, state.position)
    else
        derivatives.full_third_order!(point_data.velocity_update_data.derivs[1], state.position)
        derivatives.hessian!(point_data.velocity_update_data.value, state.position)
    end


    derivatives.gradient!(point_data.gradient, state.position)
    
    nothing
end

function fetch_core_data!(pdmp::PDMP, evo_data::LagrangianEvoData, numerics::NumericalParameters, state::SplitState, dyn::VelocityODE)
    #We determine the values of the Hessian, its Jacobian, and other relevant data at the point X.
    fetch_point_data!(evo_data.core.point_data, pdmp, state, numerics.derivatives, dyn)
    
    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = evo_data.core.point_data.velocity_update_data.value
    fetch_spectral_data!(evo_data.core.spectral_data, hessian, pdmp.method.hardness)

    fetch_evo_tensors!(evo_data, pdmp.target.dimension)
    nothing
end

function fetch_core_data!(pdmp::PDMP{Version6_2, F, D, T}, evo_data::LagrangianEvoData, numerics::NumericalParameters, state::SplitState, transition::GradientReflection) where {F, D, T}
    pd = evo_data.core.point_data
    sd = evo_data.core.spectral_data
    
    #We determine the values of the Hessian and the gradient (stored in *velocity* data here)
    numerics.derivatives.hessian!(pd.velocity_update_data.value, state.position)

    numerics.derivatives.gradient!(pd.gradient, state.position)

    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = pd.velocity_update_data.value
    fetch_spectral_data!(sd, hessian, pdmp.method.hardness)

    nothing
end

function default_grad(log_density)
    return (grad, X) -> ForwardDiff.gradient!(grad, log_density, X)
end

function default_hessian(log_density)
    return (hess, X) -> ForwardDiff.hessian!(hess, log_density, X)
end

function default_full_3rd_order(log_density)
    hf(X) = ForwardDiff.hessian(log_density, X)
    dhf!(velocity_update_data, X) = (velocity_update_data = ForwardDiff.jacobian!(velocity_update_data, hf, X))
    return dhf!
end

function default_directional_3rd_order(log_density)
    hf(X, v) = (t -> ForwardDiff.hessian(log_density, (X .+ (t .* v))))
    vhf!(position_update_data, X, v) = (position_update_data = ForwardDiff.derivative!(position_update_data, hf(X, v), 0.0))
    return vhf!
end

function default_derivatives(pdmp::PDMP{M, F, D, T}) where {M<:Lagrangian_Method,F,D,T}
    return LagrangianDerivatives(
            default_grad(pdmp.target.log_density), 
            default_hessian(pdmp.target.log_density), 
            default_full_3rd_order(pdmp.target.log_density), 
            default_directional_3rd_order(pdmp.target.log_density),
            true
            )
end
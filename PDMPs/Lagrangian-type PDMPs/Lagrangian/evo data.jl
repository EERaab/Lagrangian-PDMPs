mutable struct EvoTensorsLagrangian<:EvoTensors
    R2::Array{Float64, 3}
    R0::Array{Float64, 1}
    metric::Array{Float64, 2}
    christoffel_trace::Array{Float64, 1}
    QTHQ::Array{Float64, 3}
    form::Array{Float64, 3}
end

function initialize_evo_tensors(pdmp::PDMP{Lagrangian})
    dim = pdmp.target.dimension
    return EvoTensorsLagrangian(zeros(Float64, dim, dim, dim), zeros(Float64, dim), zeros(Float64, dim, dim), zeros(Float64, dim), zeros(Float64, dim, dim, dim), zeros(Float64, dim, dim, dim))
end

function fetch_evo_tensors!(method::Lagrangian, evolution_data::LagrangianEvoData, dyn::VelocityODE)
    specdata = evolution_data.spectral_data
    pointdata = evolution_data.point_data
    Q = specdata.Q
    QT = specdata.Q'
    J = specdata.jmatrix
    Dinv = specdata.Dinv
    evo_tensors = evolution_data.evo_tensors
    jachess_tens = pointdata.velocity_update_data.derivs[1]
    
    #We have implemented the functions using the TensorOperations package. 
    #It is not necessarily the fastest, but certainly very convenient and comparatively elegant.

    #We construct the array (Q^T H_{c} Q)_{ab}
    @tensor evo_tensors.QTHQ[a,b,c] = QT[a,k]*jachess_tens[k,l,c]*Q[l,b]

    #A quirk of Julia implies that M_{abc} = (J ∘ (Q^T H_{c} Q))_{ab} can be computed as follows;
    M = J .* evo_tensors.QTHQ 
    QDQT = (Q*Dinv*QT)

    #A particular factor - Dinv_{ak}*M_{kbc} - appears in several distinct places here. 
    @tensor evo_tensors.form[a,b,c] = Dinv[a,k]*M[k,b,c]

    #The trace is the Christoffel symbol trace
    @tensor evo_tensors.christoffel_trace[a] = evo_tensors.form[k,k,a]/2

    #The R0 term is necessary for Lagrangian velocities (∇π - tr(Γ)) 
    evo_tensors.R0 .= QDQT * (pointdata.gradient - evo_tensors.christoffel_trace)

    evo_tensors.metric .= Q*inv(Dinv)*QT

    #The R2 term
    @tensor evo_tensors.R2[a,b,c] = (-Q[a,k]*evo_tensors.form[k,m,b]*QT[m,c]) + (QDQT[a,k]*Q[b,l]*M[l,m,k]*QT[m,c]/2)

    nothing
end

#For ForwardDiff we use this function to fetch point data.
function fetch_point_data!(point_data::PointData, pdmp::PDMP{Lagrangian, F, D, T}, state::SplitState, diff_method::ForwardDer, dyn_type::VelocityODE) where {F,D,T}
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
function fetch_point_data!(point_data::PointData, pdmp::PDMP{Lagrangian, F, D, T}, state::SplitState, diff_method::AnalyticalDer, dyn_type::VelocityODE) where {F, D, T}
    diff_method.gradient!(point_data.gradient, state.position)

    diff_method.hessian!(point_data.velocity_update_data.value, state.position)

    diff_method.third_order_full!(point_data.velocity_update_data.derivs[1], state.position)

    nothing
end

function fetch_evo_data!(pdmp::PDMP{Lagrangian, F, D, T}, evo_data::LagrangianEvoData, numerics::NumericalParameters, state::SplitState, dyn::VelocityODE) where {F, D, T}
    #We determine the values of the Hessian, its Jacobian, and other relevant data at the point X.
    fetch_point_data!(evo_data.point_data, pdmp, state, numerics.diff_method, dyn)
    
    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = evo_data.point_data.velocity_update_data.value
    fetch_spectral_data!(evo_data.spectral_data, hessian, pdmp.method.hardness)

    fetch_evo_tensors!(pdmp.method, evo_data, dyn)
    nothing
end


function fetch_rates!(rates::MVector{N, Float64}, pdmp::PDMP{Lagrangian, F, D, T}, state::SplitState, evolution_data::LagrangianEvoData, 
    numerics::LagrangianNumerics, dyn::PositionVelocity; reverse::Bool = false, reversed_pdmp::Bool = false, adaptive_fwd::Bool = false) where {N, F, D, T}
    #We determine the values of the Hessian, its Jacobian, and other relevant data at the point X.
    fetch_point_data!(evolution_data.point_data, pdmp, state, numerics.diff_method, dyn)
    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = evolution_data.point_data.position_update_data.value
    fetch_spectral_data!(evolution_data.spectral_data, hessian, pdmp.method.hardness)

    #We determine the value of rho which fully determines the rate.
    gr = dot(evolution_data.point_data.gradient, state.auxiliary)
    #gr = evolution_data.point_data.gradient'*state.auxiliary
    k = rho_point_value(evolution_data, gr, state.auxiliary)
    
    #Finally we return the apropriate rate, depending on our pdmp and possible reversal.
    if (reversed_pdmp && reverse)||(!reversed_pdmp && !reverse)
        #We update the acutal rate.
        rates[1] = max(0.0, k)
    else
        rates[1] = max(0.0, -k)
    end
    if adaptive_fwd
        evolution_data.adaptive_data.fwd_rho[1] = k
    else
        evolution_data.adaptive_data.rho[1] = k
    end
    nothing
end

function rho_point_value(evo_data::LagrangianEvoData, gr::Float64, v::Vector{Float64})::Float64
    specdata = evo_data.spectral_data
    pointdata = evo_data.point_data
    Q = specdata.Q
    QT = specdata.Q'
    J = specdata.jmatrix
    Dinv=specdata.Dinv

    dirhess = DiffResults.derivative(pointdata.position_update_data)
    trash1 = evo_data.trash_matrix1
    trash2 = evo_data.trash_matrix2
    mul!(trash1, dirhess, Q)
    mul!(trash2, QT, trash1)
    trash1 .= J .* trash2

    P = evo_data.trash_vec
    mul!(P, QT, v)
    term1 = dot(P,trash1,P)/2.0
    
    #term2 = -tr(trash1*Dinv)/2.0
    term2 = 0.0
    @inbounds for i in eachindex(Dinv.diag)
        term2 -= trash1[i,i] * Dinv.diag[i] 
    end
    term2 /= 2.0
    return term1+term2-gr
end

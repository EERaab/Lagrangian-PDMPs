function fetch_rates!(rates::MVector{N, Float64}, pdmp::PDMP{Lagrangian, F, D, T}, state::SplitState, evolution_data::EvolutionData, 
    numerics::NumericalParameters, dyn::PositionVelocity; reverse::Bool = false, reversed_pdmp::Bool = false, adaptive_fwd::Bool = false) where {N, F, D, T}
    #We determine the values of the Hessian, its Jacobian, and other relevant data at the point X.
    fetch_point_data!(evolution_data.core.point_data, pdmp, state, numerics.derivatives, dyn)
    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = evolution_data.core.point_data.position_update_data.value
    fetch_spectral_data!(evolution_data.core.spectral_data, hessian, pdmp.method.hardness)

    #We determine the value of rho which fully determines the rate.
    gr = dot(evolution_data.core.point_data.gradient, state.auxiliary)
    #gr = evolution_data.point_data.gradient'*state.auxiliary
    ρ = rho_point_value(evolution_data, gr, state.auxiliary)

    #Finally we update the apropriate rate, depending on our pdmp and possible reversal.
    double_reversal = (reversed_pdmp && reverse)||(!reversed_pdmp && !reverse)
    sgnd_rho = double_reversal ? -ρ : ρ
    rates[1] = max(0.0, sgnd_rho)
    if adaptive_fwd
        evolution_data.workspace.adaptive_data.fwd_rho[1] = sgnd_rho
    else
        evolution_data.workspace.adaptive_data.rho[1] = sgnd_rho
    end
    nothing
end

function rho_point_value(evo_data::EvolutionData, gr::Float64, v::Vector{Float64})::Float64
    specdata = evo_data.core.spectral_data
    pointdata = evo_data.core.point_data
    Q = specdata.Q
    QT = specdata.Q'
    J = specdata.jmatrix
    Dinv = specdata.Dinv

    dirhess = DiffResults.derivative(pointdata.position_update_data)
    trash1 = evo_data.workspace.matrix1
    vM = evo_data.workspace.matrix2 #vM = (QT*v(H)*Q) as for velocities - just with an extra v-contraction
    mul!(trash1, dirhess, Q)
    mul!(vM, QT, trash1)
    vL = trash1
    vL .= J .* vM #vL = J ∘ vM as for velocities

    P = evo_data.workspace.vector1
    mul!(P, QT, v)
    term1 = -dot(P,vL,P)/2.0 #G_{ij,k}v^iv^jv^k 
    
    term2 = 0.0
    @inbounds for i in eachindex(Dinv.diag)
        term2 += vL[i,i] * Dinv.diag[i] 
    end
    term2 /= 2.0
    return term1+term2+gr
end

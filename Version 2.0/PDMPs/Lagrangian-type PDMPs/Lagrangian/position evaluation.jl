function fetch_rates!(rates::MVector, pdmp::PDMP{Lagrangian}, state::SplitState, evolution_data::LagrangianEvoData{EvoTensorsLagrangian}, numerics::LagrangianNumerics, dyn::PositionVelocity; reverse::Bool = false, adaptive::Bool = false)
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
    if (pdmp.reversed && reverse)||(!pdmp.reversed && !reverse)
        #We update the acutal rate.
        rates[1] = max(0, k)
        #If we use adaptive methods we will adapt not based on the rates but on the "signed" rates, which must be returned
        if adaptive
            return k
        end
        return rates
    end
    #If we use adaptive methods we will adapt not based on the rates but on the "signed" rates, which must be returned
    rates[1] = max(0, -k)
    if adaptive 
        return -k
    end
    return rates
end

function rho_point_value(evoldata::LagrangianEvoData{EvoTensorsLagrangian}, gr::Float64, v::Vector{Float64})::Float64
    specdata = evoldata.spectral_data
    pointdata = evoldata.point_data
    Q = specdata.Q
    QT = specdata.Q'
    J = specdata.jmatrix
    Dinv=specdata.Dinv

    P = QT*v
    dirhess = DiffResults.derivative(pointdata.position_update_data)
    vM = J .* (QT*dirhess*Q)
    term1 = dot(P,vM,P)/2.0
    term2 = -tr(vM*Dinv)/2.0
    return term1+term2-gr
end

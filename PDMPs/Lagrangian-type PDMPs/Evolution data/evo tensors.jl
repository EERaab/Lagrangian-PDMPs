mutable struct EvoTensors 
    #These fields are necessary to determine the rates and flow
    Γ_trace::Vector{Float64}
    Γ::Array{Float64, 3} #Unfortunately we don't have a struct for totally symmetric objects
    metric::Symmetric{Float64, Matrix{Float64}}
    metric_inv::Symmetric{Float64, Matrix{Float64}}
    grad_log_pi::Vector{Float64}
    #These fields are just here to avoid allocations
end

function fetch_evo_tensors!(method::PDMP_Method, evolution_data::LagrangianEvoData, dim::Int64)
    specdata = evolution_data.spectral_data
    pointdata = evolution_data.point_data

    Q = specdata.Q
    QT = specdata.Q'
    J = specdata.jmatrix
    Dinv = specdata.Dinv

    evo_tensors = evolution_data.evo_tensors
    Γ_trace = evo_tensors.Γ_trace
    Γ = evo_tensors.Γ #Γ^{a}_{bc} = Γ[a,b,c]
    G = evo_tensors.metric
    G_inv = evo_tensors.inverse_metric
    ∇π = evo_tensors.gradient

    jachess_tens = pointdata.velocity_update_data.derivs[1] #i.e. ∂_i ∂_j ∂_k log π 

    trash_matrix1 = evolution_data.trash_matrix1
    trash_matrix2 = evolution_data.trash_matrix2
    #The object M[i,j,k] ∼ (M_k)_{ij} = (Q^T (∂_k H) Q)_{ij} needs to be computed to determine derivatives.
    M = evo_tensors.M
    @inbounds for i in 1:dim
        @views mul!(trash_matrix1, jachess_tens[:,:,i], Q)
        @views mul!(M[:,:,i], QT, trash_matrix1)
    end

    #The object L[i,j,k] ∼ L_{ijk} = (J ∘ M_k)_{ij} is used in every calculation
    L = evo_tensors.L
    #By a quirk of broadcasting in Julia we get L[i,j,k] ∼ L_{ijk} = (J ∘ M_k)_{ij} ∼ J .* M 
    L .= J .* M


    #If we were only interested in the dynamics it would not be necessary to compute Γ fully,
    #we could instead for symmetry by contraction! Alas, we shall compute Γ fully here. 
    #This will amount to a use using and extra symmetrization.
    #To clarify:
    #Γ^i_{jk} = G^{il}((G_{lj,k}+G_{lk,j})-G_{jk,l})/2
    #so letting
    #Term1[i,j,k] ∼ Term1_{ijk} = G^{il}G_{lj,k}
    #Term2[i,j,k] ∼ Term2_{ijk} = G^{il}G_{jk,l}
    #we have
    #Γ^{i}_{jk} ∼ (Term1{ijk} + Term1{ikj} - Term2_{ijk})/2

    #The first term admits a computational simplification for the soft-abs metric.
    #Term1[i,j,k] ∼ G^{il}G_{lj,k} = (Q≀D≀^{-1} (J\circ M_k) Q^T)_{ij} =(Q ≀D≀^{-1} L_k Q^T)_{ij} ∼ 
    #∼(Q*≀D≀^{-1}*L[:,:,k]*Q^T)[i,j]

    #We can get the metric inverse while were computing Term1
    mul!(trash_matrix1, Q, Dinv)
    mul!(G_inv, trash_matrix1, QT)

    #Term1[i,j,k] ∼ (Q*≀D≀^{-1}*L[:,:,k]*Q^T)[i,j]
    Term1 = evo_tensors.Term1
    @inbounds for k in 1:dim
        @views mul!(trash_matrix2, L[:,:,k], QT)
        @views mul!(Term1[:,:,k], trash_matrix1, trash_matrix2)
    end

    #Term2[i,j,k] ∼ G^{il}G_{jk,l} = ∑_l(Q≀D≀^{-1}Q^T)_{il}(Q * L* Q)
    Term2 .*= 0.0
    @inbounds for i in 1:dim
        @views mul!(Term2[i,i,:], )
        for j in i+1:dim








    Γ[a, b, c] = 
    @inbounds for i in 1:dim
        mul!(trash_matrix, )
        mul!(∂_iG_jk[i, :, :]) 
    end

    #We construct the array (Q^T H_{c} Q)_{ab} = (M_{c})_{ab} = M_[a b c]
    @tensor M[a,b,c] = QT[a,k]*jachess_tens[k,l,c]*Q[l,b]


    #A quirk of Julia implies that M_{abc} = (J ∘ (Q^T H_{c} Q))_{ab} can be computed as follows;
    L = J .* evo_tensors.QTHQ 
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
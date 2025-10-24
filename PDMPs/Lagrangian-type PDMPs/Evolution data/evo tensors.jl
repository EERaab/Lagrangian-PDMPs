struct EvoTensors 
    #These fields are necessary to determine the rates and flow
    Γ_trace::Vector{Float64}
    Γ::Array{Float64, 3} #Unfortunately we don't have a struct for totally symmetric objects
    metric::Symmetric{Float64, Matrix{Float64}}
    metric_inv::Symmetric{Float64, Matrix{Float64}}
    gradient::Vector{Float64}
    #These fields correspond to the two types of term in Γ^i_{jk}.
    Term1::Array{Float64, 3} 
    Term2::Array{Float64, 3} 
    #These fields are just here to avoid allocations
    trash_r3_tensor1::Array{Float64, 3} 
    trash_r3_tensor2::Array{Float64, 3} 

    function EvoTensors(dim::Integer)
        Γ_trace = zeros(Float64, dim)
        Γ = zeros(Float64, dim, dim, dim)
        metric = Symmetric(zeros(Float64, dim, dim))
        metric_inv = Symmetric(zeros(Float64, dim, dim))
        gradient = zeros(Float64, dim)
        #These fields correspond to the two types of term in Γ^i_{jk}.
        Term1 = zeros(Float64, dim, dim, dim)
        Term2= zeros(Float64, dim, dim, dim)
        #These fields are just here to avoid allocations
        trash_r3_tensor1 = zeros(Float64, dim, dim, dim)
        trash_r3_tensor2 = zeros(Float64, dim, dim, dim)
        return new(Γ_trace, Γ, metric, metric_inv, gradient, Term1, Term2, trash_r3_tensor1, trash_r3_tensor2)
    end
end

function diagonal_inv!(D::Diagonal)
    D.diag .= D.diag .^(-1)
    return D
end

function simplified_trace(D::Diagonal, M)
    S = 0.0
    @inbounds for i in eachindex(D.diag)
        S += D.diag[i] * M[i,i]
    end
    return S
end

function fetch_evo_tensors!(evolution_data::EvolutionData, dim::Int64)
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
    G_inv = evo_tensors.metric_inv
    #This is goofy but we do not want to feed the full evo_data into dynamics so we save the gradient in two places.
    evo_tensors.gradient .= pointdata.gradient 

    jachess_tens = pointdata.velocity_update_data.derivs[1] #i.e. ∂_i ∂_j ∂_k log π 

    trash_matrix1 = evolution_data.trash_matrix1
    trash_matrix2 = evolution_data.trash_matrix2
    #The object M[i,j,k] ∼ (M_k)_{ij} = (Q^T (∂_k H) Q)_{ij} needs to be computed to determine derivatives.
    M = evo_tensors.trash_r3_tensor1 #evo_tensors.M
    @inbounds for k in 1:dim
        @views mul!(trash_matrix1, jachess_tens[:,:,k], Q)
        @views mul!(M[:,:,k], QT, trash_matrix1)
    end

    #The object L[i,j,k] ∼ L_{ijk} = (J ∘ M_k)_{ij} is used in almost every calculation below
    L = evo_tensors.trash_r3_tensor2
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
    mul!(G_inv.data, trash_matrix1, QT)

    #Term1[i,j,k] ∼ (Q*≀D≀^{-1}*L[:,:,k]*Q^T)[i,j]

    #Term2[i,j,k] = ∑_l G_inv[i,l](Q * L[:,:,l]* QT)[j,k]
    #For convenience we store S[:,:,l] = (Q * L[:,:,l]* QT) = (∂_l G)[:,:]
    #Thus Term2[i,j,k] = ∑_l S[i,j,l]*G_inv[l,k] (and S is symmetric in i,j).
    S = evo_tensors.trash_r3_tensor1
    #partial_trace_vec = evo_tensors.trash_vector #components 
    #S is of course just the same as M at this point, but M is no longer needed.
    Term1 = evo_tensors.Term1
    @inbounds for k in 1:dim
        @views mul!(trash_matrix2, L[:,:,k], QT)
        @views mul!(Term1[:,:,k], trash_matrix1, trash_matrix2)
        @views mul!(S[:,:,k], Q, trash_matrix2)
        @views Γ_trace[k] = simplified_trace(Dinv, L[:,:,k]) #this admits a simpler form.
    end

    #Term2[i,j,k] ∼ G^{il}S_{jk,l}
    Term2 = evo_tensors.Term2
    @inbounds for i in 1:dim
        @views mul!(Term2[:,i,i], G_inv, S[i,i,:])
        for j in i+1:dim
            @views mul!(Term2[:,i,j], G_inv, S[i,j,:])
            @views Term2[:,j,i] .= Term2[:,i,j]
        end        
    end

    @inbounds for i in 1:dim
        @views Γ[i,:,:] .= (Term1[i,:,:] .+ (Term1[i,:,:]') .- Term2[i,:,:])./2.0
    end

    #Now for the metric
    #Notably this maps Dinv ↦ D so Dinv no longer stores Dinv actually.
    D = diagonal_inv!(Dinv) 
    mul!(trash_matrix1, D, QT)
    mul!(G.data, Q, trash_matrix1)
    #If we wanted to we could restore Dinv:
    #diagonal_inv!(Dinv)
    return evo_tensors
end
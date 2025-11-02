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

function multiply_by_diag_inv!(trash_matrix, diag_matrix::Diagonal, right_matrix)
    for i in eachindex(diag_matrix.diag)
        @views trash_matrix[i,:] .=  right_matrix[i,:] ./ diag_matrix.diag[i]
    end
    return trash_matrix
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

    #JH_{ijk} =∂_i ∂_j ∂_k log π i.e. jachess_tens[i,j,k]. Totally symmetric.
    jachess_tens = pointdata.velocity_update_data.derivs[1] 

    trash_matrix1 = evolution_data.trash_matrix1
    trash_matrix2 = evolution_data.trash_matrix2    

    #(M_k)_{ij} = (Q^T*H_k*Q)_{ij} i.e. M[i,j,k] = (QT * H[:,:,k]*Q)[i,j]
    M = evo_tensors.trash_r3_tensor1 



    for k in 1:dim
        @views mul!(trash_matrix1, jachess_tens[:,:,k], Q)
        @views mul!(M[:,:,k], QT, trash_matrix1)
    end
    #(L_k)_{ij} = (J ∘ M_k)_{ij} i.e. L[i,j,k] = J[i,j]*M[i,j,k] = (Julia broadcasting quirk) = (J .* M)[i,j,k]
    L = evo_tensors.trash_r3_tensor2
    L .= J .* M

    #Γ^i_{jk}= (1/2)*G^{il}(G_{lj,k}+G_{lk,j}-G_{jk,l})
    #With
    #T1^i_{jk} = G^{il}G_{lj,k} i.e. T1[i,j,k] = ((Q*≀D≀^{-1}*QT)(Q*L[:,:, k]*QT))[i,j] = (Q*≀D≀^{-1}*L[:,:,k]*QT)[i,j]
    #and
    #T2^i_{jk} = G^{il}G_{jk,l} i.e. T2[i,j,k] = ∑_L  (Q*≀D≀^{-1}*QT)[i,l] (Q L[:,:,l] QT)[j,k]
    #we have
    #Γ[i,j,k] = (1/2)(T1[i,:,:]+T1[1,:,:]' - T2[i,:,:])[j,k]

    #Let TrashM1 = (Q*≀D≀^{-1})
    #Then G_inv = trash_matrix1*QT
    mul!(trash_matrix1, Q, Dinv)
    mul!(G_inv.data, trash_matrix1, QT)

    #Let S[i,j,k] = (Q*L[:,:,k]*QT)[i,j] so that T2[i,j,k] = ((Q*≀D≀^{-1}*QT)*S[j,k,:])[i]
    #also
    #T2[:, j, k] = ((Q*≀D≀^{-1}*QT)*S[j,k,:])

    S = evo_tensors.trash_r3_tensor1
    T1 = evo_tensors.Term1
    for k in 1:dim
        @views mul!(trash_matrix2, L[:,:,k], QT)
        @views mul!(T1[:,:,k], trash_matrix1, trash_matrix2)
        @views mul!(S[:,:,k], Q, trash_matrix2)
        @views Γ_trace[k] = simplified_trace(Dinv, L[:,:,k])/2.0 #this admits a simpler form.
    end

    T2 = evo_tensors.Term2
    for j in 1:dim
        @views mul!(T2[:,j,j], G_inv, S[j,j,:])
        for k in j+1:dim
            @views mul!(T2[:,j,k], G_inv, S[j,k,:])
            @views T2[:,k,j] .= T2[:,j,k]
        end        
    end

    for i in 1:dim
        @views Γ[i,:,:] .= (T1[i,:,:] .+ (T1[i,:,:]') .- T2[i,:,:])./2.0
    end

    #Now for the metric
    multiply_by_diag_inv!(trash_matrix1, Dinv, QT)
    mul!(G.data, Q, trash_matrix1)

    ##OLD METHOD
    #Notably this maps Dinv ↦ D so Dinv no longer stores Dinv actually.
        #D = diagonal_inv!(Dinv) 
        #mul!(trash_matrix1, D, QT)
        #mul!(G.data, Q, trash_matrix1)
        #If we wanted to we could restore Dinv:
        #diagonal_inv!(Dinv)
        #@show U .≈ G
    return evo_tensors
end
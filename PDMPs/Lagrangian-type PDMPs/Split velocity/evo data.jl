
struct SplitVelocityData
    velocity_partition::BinaryMinHeap{Tuple{Float64, Int64}}
    reduced_v::Vector{Float64}
    signed_rho_J::Matrix{Float64} #Could be tuple of static vectors!
    As::Matrix{Float64} #Could be tuple of statics!
    sign_vector::Vector{Int64}
    M_adj_rhoJ:: Matrix{Float64} #Could be static matrix!
    #M_matrix::MMatrix{4, 4, Float64} #Could be static matrix!
    #F_positive::MVector{4, Float64}
    #F_negative::MVector{4, Float64}

    function SplitVelocityData(dim)
        velocity_partition = BinaryMinHeap{Tuple{Float64, Int64}}()
        reduced_v = zeros(Float64, dim)
        signed_rho_J = zeros(Float64, dim + 1, 4)
        As = zeros(Float64, dim + 1, 4)
        sign_vector = zeros(Float64, dim + 1)
        M_adj_rhoJ = zeros(Float64, dim + 1, 4)
        #M_matrix = @MMatrix zeros(Float64, 4, 4)
        return new(velocity_partition, reduced_v, signed_rho_J, As, sign_vector, M_adj_rhoJ)#, M_matrix)
    end
end

struct SplitEvoData{S, PD, T, N}
    segments::S
    core::LagrangianCoreData{PD, T}
    workspace::LagrangianWorkspaceVariables{N}
    split_data::SplitVelocityData

    function SplitEvoData(dim::Integer)
        segments = Dict{Int64, Segment{dim}}((dim) => Segment{dim}())
        core = LagrangianCoreData(dim)
        workspace = LagrangianWorkspaceVariables(dim)
        split_data = SplitVelocityData(dim)
        return new{typeof(segments), typeof(core.point_data), typeof(core.evo_tensors), dim}(segments, core, workspace, split_data)
    end

    function SplitEvoData(pdmp::PDMP)
        return SplitEvoData(pdmp.target.dimension)
    end
end


#This method is exactly analogous to the method for Lagrangian or Version6.2
function fetch_core_data!(pdmp::PDMP, evo_data::SplitEvoData, numerics::NumericalParameters, state::SplitState, dyn::SplitVelocity)
    #We determine the values of the Hessian, its Jacobian, and other relevant data at the point X.
    fetch_point_data!(evo_data.core.point_data, pdmp, state, numerics.derivatives, dyn)

    #The point data is processed through an eigendecomposition, which is used to define the spectral data (Q, QT, Dinv, J).
    hessian = evo_data.core.point_data.velocity_update_data.value
    fetch_spectral_data!(evo_data.core.spectral_data, hessian, pdmp.method.hardness)

    fetch_evo_tensors!(evo_data, pdmp.target.dimension)
    nothing
end



#Computing the objects A^{(n)}_{I;J} for I ≠ J. Stored as a matrix, As[I, n] = A^{(n)}_{I;J}.
function fetch_divergences!(dim, J, evo_data)
    A = evo_data.split.As
    reduced_v = evo_data.split.reduced_v

    #reduced_v .= velocity
    #reduced_v[J] = 0.0

    Γ = evo_data.core.evo_tensors.Γ
    G = evo_data.core.evo_tensors.metric
    G_inv = evo_data.core.evo_tensors.metric_inv
    ∇φ = error("Not properly implemented yet -throwing error to avoid troubles later")#evo_data.core.evo_tensors.gradient - Ginv*Γ_trace or something like that.
    
    @views A[dim+1, :] .= 0.0

    @inbounds for I in 1:dim
        @views A[I, 4] = -Γ[I, J, J] * G[I, J]
        
        A[dim+1, 4] -= A[I, 4]

        @views A[I, 3] = -Γ[I, J, J] * dot(G[I, :], reduced_v) - 2 * G[I, J] * dot(Γ[I, J, :], reduced_v)

        A[dim+1, 3] -= A[I, 3]

        @views A[I, 2] = 2*Γ[I, I, J] - 2*dot(Γ[I, J, :], reduced_v)*dot(G[I, :], reduced_v) - G[I, J] * dot(reduced_v, Γ[I, :, :], reduced_v) - G[I, J] * dot(G_inv[I, :], ∇φ)

        A[dim+1, 2] -= A[I, 2]

        @views A[I, 1] = 2*dot(Γ[I, I, :], reduced_v) - dot(reduced_v, Γ[I, :, :], reduced_v)*dot(G[I, :], reduced_v) - dot(G[I, :], reduced_v) * dot(G_inv[I, :], ∇φ)

        A[dim+1, 1] -= A[I, 1]
    end

    return A
end

function fetch_signed_rho!(ρJ, J, A, dim; reversed_pdmp = false)
    #We should transpose ρJ as Julia uses column-major order
    @inbounds for I in 1:dim
        if I ≠ J
            @views ρJ[I,:] .= A[J,:] .- A[I,:]
        else
            @views ρJ[J,:] .= 0.0
        end
    end

    @views ρJ[n+1, :] .= (-A[n+1,:] ./ dim) 

    if reversed_pdmp
        return -ρJ
    end
    return ρJ
end


function fetch_velocity_parameters!(evo_data::SplitEvoData, J; reversed_pdmp::Bool = false)

    vred = evo_data.split_data.reduced_v
    evo_tensors = evo_data.core.evo_tensors
    Γ = evo_tensors.Γ
    Ginv = evo_tensors.metric_inv
    ∇φ = error("Not properly implemented yet -throwing error to avoid troubles later")

    a = -Γ[J,J,J] 

    @views b = -2*dot(Γ[J, J,:], vred)

    @views c = - dot(Ginv[J,:], ∇φ) - dot(vred, Γ[J,:,:], vred)

    if reversed_pdmp
        return (-a,-b,-c)
    end
    return (a,b,c)
end

include("split flow functions.jl")


function fetch_split_data!(pdmp::PDMP, evo_data::SplitEvoData, numerics, state, dyn::SplitVelocityFlow; reversed_pdmp::Bool = false)
    split_data = evo_data.split_data
    As = split_data.As
    v_red = split_data.reduced_v
    J = dyn.component
    dim = pdmp.target.dimension
    ρJ = split_data.signed_rho_J
    M_adj_ρJ = split_data.M_adj_rhoJ

    #To compute our contractions we set v_red here:
    v_red .= state.auxiliary
    v_red[J] = 0.0

    #We compute the I-divergences:
    fetch_divergences!(dim, J, evo_data)

    #Note that as for the other Lagrangian PDMPs we define ρ explicitly as a particular combination 
    #of tensors. Under reversal of the PDMP we do not redefine ρ, rather we return ρ_sgnd = - ρ.
    #In other words: ρ_sgnd = (-1)^reversed_pdmp × ρ.
    #Here this is implemented in 'fetch velocity parameters' and in the value of 'ρJ'.

    #We compute ρJ_sgnd from which we can determine the rates:
    fetch_signed_rho!(ρJ, J, As, dim, reversed_pdmp = reversed_pdmp) 


    #We need to build F-vectors. To do this we take several stepes
        #To determine the initial rates we need the M-matrices.
        #These can only be derived once the dynamic is computed:
        (a,b,c) = fetch_velocity_parameters!(evo_data, J, reversed_pdmp = reversed_pdmp) 
        (dir, param, flow_class) = direction_velocity_and_type(a, b, c, state.auxiliary[J])

        #Now we compute the M-matrix:
        α, β = M_matrix_parameters(flow_class, param)
        M_matrix = static_M_matrix!(α, β)

        #We build the M-adjusted ρJ:
        #OPTIMIZE: Build the transpose instead of M to avoid transposition & consider column-major
        mul!(M_adj_ρJ, M_matrix', ρJ)
        
        #From the ρJ we compute the F-vectors and sign vectors.
        ini_u_tuple = (u0^3, u0^2, u0, 1.0) 
        fetch_sign_vector!(evo_data, ini_u_tuple, J)
        
        #Finally we compute the F-vectors.
        F_positive, F_negative = fetch_F_vectors!(evo_data, J)
    ####
    
    return (dir, param, flow_class, F_positive, F_negative)
end


function fetch_sign_vector!(evo_data::SplitEvoData, u0_tuple, J)
    sign_vector = evo_data.sign_vector #i.e. the initial rates
    M_adj_ρJ = evo_data.M_adj_rhoJ
    for I in eachindex(sign_vector)
        if I ≠ J
            sign_vector[I] = sgn(dot(@view(M_adj_ρJ[I, :]), u0_tuple))
        else
            sign_vector[J] = 0
        end
    end

    return sign_vector
end

function fetch_F_vectors!(evo_data, J)
    split_data = evo_data.split_data
    F_positive = @SVector zeros(Float64,4)#split_data.F_positive 
    F_negative = @SVector zeros(Float64,4)#split_data.F_negative
    M_adj_rho = split_data.M_adj_rhoJ

    for I in eachindex(sign_vector)
        if iszero(sign_vector[I]) 
            if !(I == J)
                #Implement a function that checks the behaviour of ρ_{JI}. For now we throw an error
                error("Unimplemented! But should be implemented")
            end
        elseif sign_vector[I] > 0
            @views F_positive += M_adj_rho[I,:]
        else 
            @views F_negative += M_adj_rho[I,:]
        end
    end
    return F_positive, F_negative
end


function M_matrix_parameters(flow_class::VelocityFunctions, param)::Tuple{Float64, Float64}
    # This is a bit goofy, but should be OK.
    if flow_class == type0
        error("This should not be called.")
    elseif flow_class == type1
        α = param[2]*param[3]
        β = param[1]
    elseif flow_class == type2
        α = -param[2]/param[3]
        β = param[3]
    elseif flow_class == type3
        α = -param[4]
        β = param[1]
    elseif flow_class == type4
        α = -param[4]*param[5]
        β = param[1]*param[2]
    elseif flow_class == type5
        α = -param[2]
        β = param[1]
    else
        error("Flow type recognition error.")
    end
    return α, β
end

function static_M_matrix!(α::Float64, β::Float64)
    M = @SMatrix [
            (β^3)   3.0*α*(β^3) 3.0*(α^2)*(β^3) 1.0*(α^3)*(β^3); 
            0.0     (β^2)       2.0*α*(β^2)     1.0*(α^2)*(β^2); 
            0.0     0.0         β               1.0*α*β; 
            0.0     0.0         0.0             1.0
            ]    
    return M
end
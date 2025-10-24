#### VELOCITY DYNAMICS ####
    """
        symmetric_double_dot(v,M)

    Returns the product v^T M v for a symmetric matrix M.
    """
    function symmetric_double_dot(v, M)
        value = 0.0
        @inbounds for i in eachindex(v)
            value += v[i]^2*M[i,i]
            for j in i+1:length(v)
                value += 2*M[i,j]*v[i]*v[j]
            end
        end
        return value
    end

    function velocity_dynamics!(du, u, p, pdmp::PDMP{M, F, D,T}, trash_vector) where {M<:Lagrangian_Method, F, D, T}
        evo_tensors = p[2]

        #Some shorthands are convenient.
        Γ_trace = evo_tensors.Γ_trace
        Γ = evo_tensors.Γ #Γ^{a}_{bc} = Γ[a,b,c]
        G = evo_tensors.metric
        G_inv = evo_tensors.metric_inv
        ∇π = evo_tensors.gradient

        #u = (ρ, ψ, v^1, v^2, …, v^n) and du[1] can be found explicitly from du[2:end]
        v = @view(u[3:end])

        du[2] = 2.0 * dot(v, Γ_trace) #dψ/dt
        
        #The velocity vector (v= du[3:end]) satisfies dv = -Γ^i_{jk}v^jv^k-G^{ij}ϕ_{,j} dt
        #For the SL-BPS method ϕ = -log π + (log det G )/2 where the latter has derivative tr(G^{-1}v(G))/2 = Γ^i_{ij}
        #For CA-BPS ϕ = (log det G )/2
        #Thus ϕ_{,i} = Γ_trace_i - β ∂_i log π with β = 0 or 1
        #Hence dv^i = -Γ^i_{jk}v^jv^k + G^{ij} (log π_{,j} - Γ^k_{kj}) dt
        if M == Lagrangian
            trash_vector .= Γ_trace .- ∇π #(log π_{,j} - Γ^k_{kj})
        elseif M == Version62
            trash_vector .= Γ_trace
        else
            error("Unimplemented!")
        end
        mul!(@view(du[3:end]), G_inv, trash_vector) 

        #Now we add v-quadratic parts of dv:
        for i in axes(Γ, 1)
            du[i+2] -= symmetric_double_dot(v, @view(Γ[i,:,:])) 
        end
        #At this point we have computed du[i] = dv^i/dt = Φ^i at v.

        #We begin to compute ρ = dψ/dt - dv^i/dt G_{ij}v^j
        #First we construct G_{ij}v^j
        mul!(trash_vector, G, v)

        du[1] = du[2] - dot(v, trash_vector)
        return du
    end


#JACOBIAN DYNAMICS
    function jacobian_dynamics!(J, u, p, pdmp::PDMP{M, F, D,T}, trash_matrix, trash_vector) where {M<:Lagrangian_Method, F, D, T}
        evo_tensors = p[2]

        #Some shorthands are convenient.
        Γ_trace = evo_tensors.Γ_trace
        Γ = evo_tensors.Γ #Γ^{a}_{bc} = Γ[a,b,c]
        G = evo_tensors.metric
        G_inv = evo_tensors.metric_inv
        ∇π = evo_tensors.gradient

        dim = pdmp.target.dimension

        #J = ∂(ρ, dψ, dv^1, dv^2, …, dv^n)/∂u^j 
        #with u = (I, ψ, v^1, …, v^n).


        #Nothing depends on ρ or ψ.
        #This is goofy. DAE instead?
        for i ∈ axes(J, 1)
            J[i, 1] = 0.0
            J[i, 2] = 0.0
        end

        #(dψ/dt)/dv = 2 ⋅ p_tr(Γ)
        @views J[2, 3:end] .= 2.0 .* Γ_trace

        #(dv/dt)/dv = -2 (Γv)
        v = @view(u[3:end])
        for j ∈ 1:dim
            @views mul!(J[j+2,3:end], (-2.0 .* Γ[j, :, :]), v)
        end

        #(dρ/dt)/dv = mess, see docs.
        @views J[1, 3:end] .= 3. .* Γ_trace

        #...but a term like J^i_jG_{il} appears
        Tmat = trash_matrix
        @views mul!(Tmat, J[3:end, 3:end]', G)
        mul!(trash_vector, Tmat, v)
        @views J[1, 3:end] .-= trash_vector
        mul!(trash_vector, Tmat', v)
        @views J[1, 3:end] .-= (trash_vector) ./ 2.0

        if M == Lagrangian
            @views J[1, 3:end] .-= ∇π    
        end
        return J
    end

#THE END
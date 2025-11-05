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

    function velocity_dynamics_split_rho!(du, u, p)
        evo_tensors = p[2]
        reversed_pdmp = p[3].x #p[3] is a ref to a bool.
        trash_vector = p[4]
        method = p[6]

        #Some shorthands are convenient.
        Γ_trace = evo_tensors.Γ_trace
        Γ = evo_tensors.Γ #Γ^{a}_{bc} = Γ[a,b,c]
        G = evo_tensors.metric
        G_inv = evo_tensors.metric_inv
        ∇π = evo_tensors.gradient

        #u = ([ρ]^+, [-ρ]^+, ψ, v^1, v^2, …, v^n) and du[1], du[2] can be found explicitly from du[3:end]
        v = @view(u[4:end])

        du[3] = 2.0 .* dot(v, Γ_trace) #dψ/dt
        
        #The velocity vector (v= du[3:end]) satisfies dv = -Γ^i_{jk}v^jv^k-G^{ij}ϕ_{,j} dt
        #For the SL-BPS method ϕ = -log π + (log det G )/2 where the latter has derivative tr(G^{-1}v(G))/2 = Γ^i_{ij}
        #For CA-BPS ϕ = (log det G )/2
        #Thus ϕ_{,i} = Γ_trace_i - β ∂_i log π with β = 0 or 1
        #Hence dv^i = -Γ^i_{jk}v^jv^k + G^{ij} (log π_{,j} - Γ^k_{kj}) dt
        if method == :Lagrangian
            trash_vector .= ∇π .- Γ_trace #(log π_{,j} - Γ^k_{kj})
        elseif method == :Version6_2
            trash_vector .= (-1) .* Γ_trace
        else
            error("Unimplemented!")
        end
        mul!(@view(du[4:end]), G_inv, trash_vector) 

        #Now we add v-quadratic parts of dv:
        for i in axes(Γ, 1)
            du[i+3] -= symmetric_double_dot(v, @view(Γ[i,:,:])) 
        end

        
        if reversed_pdmp
            du .*= (-1)
        end

        #We compute ρ = dψ/dt + dv^i/dt G_{ij}v^
        @views sgnd_ρ = du[3] + dot(du[4:end], G, u[4:end])
        du[1] = max(0.0, sgnd_ρ) #must be interpreted as the fwd rate (of the possibly reversed pdmp)
        du[2] = max(0.0, -sgnd_ρ) #rev rate 
        return du
    end


#JACOBIAN DYNAMICS
    function jacobian_dynamics_split_rho!(J, u, p)
        #This is iffy: J will be discontinuous at ρ = 0.
        evo_tensors = p[2]
        reversed_pdmp = p[3].x
        trash_vector = p[4]
        trash_matrix = p[5]
        method = p[6]
        
        #Some shorthands are convenient.
        Γ_trace = evo_tensors.Γ_trace
        Γ = evo_tensors.Γ #Γ^{a}_{bc} = Γ[a,b,c]
        G = evo_tensors.metric
        G_inv = evo_tensors.metric_inv
        ∇π = evo_tensors.gradient


        

        #J = ∂([(-1)^rev*ρ]^+, [-(-1)^rev*ρ]^+, (-1)^rev dψ, (-1)^rev dv^1, (-1)^rev dv^2, …, (-1)^rev dv^n)/∂u^j 
        #with u = (I, ψ, v^1, …, v^n).


        #Nothing depends on λ_fwd, λ_rev or ψ.
        for i ∈ axes(J, 1)
            J[i, 1] = 0.0
            J[i, 2] = 0.0
            J[i, 3] = 0.0
        end

        #(dψ/dt)/dv = 2 ⋅ p_tr(Γ)
        @views J[3, 4:end] .= 2.0 .* Γ_trace

        #(dv/dt)/dv = -2 (Γv)
        v = @view(u[4:end])
        for j ∈ eachindex(v) #to avoid referring to dim explicitly
            @views mul!(J[j+3,4:end], (-2.0 .* Γ[j, :, :]), v)
        end

        #(dρ/dt)/dv = mess, see docs.
        @views J[1, 4:end] .= Γ_trace
        if method == :Lagrangian
            @views J[1, 4:end] .+= ∇π    
        end

        #...but a term like J^i_jG_{il} appears
        Tmat = trash_matrix
        @views mul!(Tmat, J[4:end, 4:end]', G)
        mul!(trash_vector, Tmat, v)
        @views J[1, 4:end] .+= trash_vector
        mul!(trash_vector, Tmat', v)
        @views J[1, 4:end] .+= (trash_vector) ./ 2.0

        
        @views trash_vector .= J[1, 4:end]
        #At this stage trash_vector = ∂ρ/∂v^i

        #We can prove that ρ = (∂ρ/∂v^i)v^i - v^kG_{kl}(∂v^l/∂v^i)v^i ∼ dot((∂ρ/∂v),v) - dot(v, GJ[4:end, 4:end],v)
        ρ = dot(trash_vector, v) - dot(v, trash_matrix, v)
        
        #The rates are determined by the direction of the pdmp. du[1] corresponds to λ_fwd = [(-1)^pdmp_direction ρ]^+ = [sgnd_ρ]^+
        sgnd_ρ = reversed_pdmp ? -ρ : ρ

        if sgnd_ρ > 0
            #In this case the forward rate is non-zero, and has the Jacobian derivatives computed above 
            @views J[2, :] .*= 0.0
        elseif sgnd_ρ < 0
            #In this case the reverse rate is non-zero, and has the Jacobian derivatives computed above, with an extra minus-sign 
            @views J[2, :] .= -J[1, :]
            @views J[1, :] .*= 0.0
        else 
            #Should throw an error?
            @warn "Unstable derivative at ρ = 0"
        end

        #J[1,:], and J[2,:] are now unaltered, but all others can change their sign
        if reversed_pdmp
            @views J[3:end, 3:end] .*= (-1.0)
        end
        return J
    end

#THE END
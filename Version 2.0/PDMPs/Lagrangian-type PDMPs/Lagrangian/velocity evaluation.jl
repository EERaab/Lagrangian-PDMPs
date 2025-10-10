
function velocity_dynamics!(du::Union{Vector{Float64}, T}, dyn_vector::Union{Vector{Float64}, T}, pdmp::PDMP{Lagrangian}, evo_tensors::EvoTensors, dyn_type::VelocityODE{Lagrangian}) where T<:MVector
    #Some useful naming conventiongs are used
    R0 = evo_tensors.R0
    R2 = evo_tensors.R2
    metric = evo_tensors.metric
    christoffel_trace = evo_tensors.christoffel_trace
    dimension = pdmp.target.dimension

    #Old version
    #Rv = zeros(Float64, dimension)
    #for i ∈ 1:dimension, j ∈ 1:dimension
    #    Rv .+= R2[:,i,j]*dyn_vector[i+3]*dyn_vector[j+3]
    #end
    #Rv .+= R0

    #We should not have this function accept the full dyn_vector as its input.
    #Allocation free (or at least reduced) version:
    @view(du[4:end]) .= R0
    for i ∈ axes(R2, 1)
        du[i+3] += dot(@view(dyn_vector[4:end]), @view(R2[i,:,:]), @view(dyn_vector[4:end]))
    end
        
    #The flow is dv=R^L dt so under the reversal we get dv = - R^L dt
    if pdmp.reversed
        @view(du[4:end]) .*= -1
    end
    
    #The "vector" g_{ab}R^b is used to determine rho
    lowered_vector = metric*@view(du[4:end])

    rho = 0.0
    vol_factor = 0.0
    for i ∈ 1:dimension
        #div(R^L) = -2(tr(Γ) ⋅ v) is the divergence of the forward velocity flow along L. A term in this expression is
        div_RL = -2*christoffel_trace[i]*dyn_vector[i+3]

        #In rho we get a  term -g_{ai}R^a v^i and an associated term from -2Γ^a_{ai}v^i = ± div(R^L) for every i.
        #The term -g_{ia}R^av^i as added to rho
        rho += -lowered_vector[i]*dyn_vector[i+3]

        #While the lowered vector is calculated after we've adjusted for the direction, the divergence is not and its sign is adjusted.
        if pdmp.reversed
            rho += - div_RL
        else
            rho += div_RL
        end     
        #We always calculate the forward divergence, i.e. ∫div(R^L) which then gets added or subtracted into the final expressions for the reversed rate integral.
        vol_factor += div_RL
    end

    du[1] = max(0, -rho)
    du[2] = max(0, rho) 
    du[3] = vol_factor
    #for j ∈ 1:dimension
    #    du[j+3] = Rv[j]
    #end

    return du
end
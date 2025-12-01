
function do_the_thing(threshold::Float64, pdmp::PDMP, segment::Segment{N}, state::SplitState, evo_data::EvolutionData, 
    numerics::NumericalParameters, dyn::PositionVelocity; max_duration::Float64 = 0.0, reversed_pdmp::Bool = false)
    
    fetch_core_data!(evo_data, numerics) 

    (dir, param, flow_class, F_positive, F_negative) = fetch_split_data!(pdmp, evo_data, numerics, state, dyn; reversed_pdmp = reversed_pdmp) #


    t = t_prev = 0.0
    #The following are precomputed in 'initialize_velocity_evolution'
    T = evo_data.velocity_partition 
    sign_vector = evo_data.sign_vector

    while Λ < threshold
        forward_step!(segment, evo_data, threshold, state)

    end
end








function do_the_thing(dim, J, evo_data)
    T = evo_data.velocity_partition

    (dir, param, flow_type) = direction_velocity_and_type(a, b, c, u0)
    G_input_p = (flow_type, param)
    G(t; p) = evaluate_integral(p[1], t, p[2]) #G(t;p) is the rate integral at time t
        
    if flow_type.inverse ≠ :None
        partition_by_roots!(T, J, ρJ, dir, u0)
    else
        empty!(T)
    end

    t = 0.0
    t_prev = 0.0
    Λ = 0.0
    Λ_rev = 0.0
    while Λ < threshold
        if isempty(T) 
            if flow_type.inverse == :None
                #Just ceck
            #Check for termination point
            u = find_zero(G, (t_prev, t), root_method) 
        else
            (u, I) = pop!(T)
        end



        t_prev = t
        t = evaluate_inverse(V, param, u)
        G(t; p = G_input_p)
        G_prev= G(t_prev; p = G_input_p)      
        ΔG = G(t_prev; p = G_input_p) - G_prev
        
        ΔΛ = dot(F_positive, ΔG)
        if Λ + ΔΛ ≥ threshold
            method = a
            Z(t;q) = (G(t;p = q) - G_prev - threshold + Λ)
            t_new = find_zero(Z, (t_prev, t), , root_method) 
            
            break
        else
            Λ += ΔΛ
            Λ_rev -= dot(F_negative, ΔG)
            if sign_vector[I] > 0
                F_positive .-= @view(M_adj_ρ[I,:])
                F_negative .+= @view(M_adj_ρ[I,:])
            else
                F_positive .+= @view(M_adj_ρ[I,:])
                F_positive .-= @view(M_adj_ρ[I,:])
            end            
            sign_vector[I] = -sign_vector[I]
        end
        
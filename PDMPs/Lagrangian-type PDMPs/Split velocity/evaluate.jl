include("cubic roots.jl")


function evaluate_flow!(threshold::Float64, pdmp::PDMP, segment::Segment{N}, state::SplitState, evo_data::EvolutionData, 
    numerics::NumericalParameters, dyn::SplitVelocityFlow; max_duration::Float64 = 0.0, reversed_pdmp::Bool = false,
    verbose = false
    ) where N
    
    #We fetch the core data -  evo_tensors etc
    fetch_core_data!(pdmp, evo_data, numerics, state, dyn) #this shouldn't always be called!!!!! Note that velocity->velocity implies that we do not recompute derivatives!

    # We get the relevant split parameters 
    (dir, param, flow_class, F_positive, F_negative) = fetch_split_data!(pdmp, evo_data, numerics, state, dyn; reversed_pdmp = reversed_pdmp) #

    #The sign_vector is computed above, and we introduce some shorthand
        T = evo_data.split_data.velocity_partition 
        sign_vector = evo_data.split_data.sign_vector 
        J = dyn.component
        u = state.auxiliary[J]
    verbose ? println("Flow direction: $J") : nothing
    verbose ? println("Flow class: $flow_class") : nothing 
    verbose ? println("Parameters: $param") : nothing 
    verbose ? println("F-vectors; \n F+: $F_positive  \n  F-: $F_negative") : nothing 
    verbose ? println("Initial velocity: $u") : nothing 
    verbose ? println("Direction: $dir") : nothing 
    verbose ? println("Sign vector: $sign_vector") : nothing 


    #We compute the reverse rates at the initial point:
    ini_u_tuple = (u^3, u^2, u, 1.0)
    fetch_rate_vector!(segment.reverse_rates, evo_data, ini_u_tuple, J, rate_is_fwd = false)

    verbose ? (@show segment.reverse_rates) : nothing

    #We deal with the special case of a stationary velocity evolution.    
        if flow_class.inverse == :None
            verbose ? println("Stationary velocity case") : nothing
            #Special case: Stationary velocity.
            #We compute the reverse rate integral and the rates.
            segment.reverse_rate_integral = stationary_flow(threshold, F_positive, F_negative, sign_vector, ini_u_tuple)
            segment.forward_rate_integral = threshold
            fetch_rate_vector!(segment.forward_rates, evo_data, ini_u_tuple, J, rate_is_fwd = true)
            verbose ? println("Stationary velocity case") : nothing
            return false
        end

    ####

    #Setup for rate integration loop
        #We can now describe the rate integral as a function G
        #This is used to find zeros (or rather a max_value).
        #Schematically:
        #p = integration_parameters = (flow_class, param, max_value)
        G(t, p) = evaluate_integral(p[1], t, p[2]) .- p[3]
        
        # We start at t = 0.0 and set up our velocity heap 
        t = 0.0
        t_prev = 0.0
        u_prev = u   
        partition_by_roots!(J, evo_data, dir, u)
        verbose ? (@show T) : nothing

        #The termination condition for the loop is Λ = threshold
        Λ = segment.forward_rate_integral 
        Λ_rev = segment.reverse_rate_integral
        remainder = threshold - Λ
        non_terminal = (remainder > 0)
        
        #In the loop we track the next I for which ρJI flips its sign
        K = 0 
    ####
    
    M_adj_rho = evo_data.split_data.M_adj_rhoJ

    #From the loop we want to compute Λ_rev and u_final.
    while non_terminal 
        #We check for the next velocity at which to compute the rate integral.
        #If there is none we check for a termination point.
        u_prev = u
        if isempty(T)
            verbose ? println("Empty partition!") : nothing
            if !any(sign_vector .> 0)
                error("No positive forward rate.")
            end
            #Root find: Note that G is monotonically increasing at this point.
            integration_parameters = (flow_class, param, remainder)
            u = find_zero(G, u, numerics.auxiliary_method, integration_parameters)
            non_terminal = false
        else
            #We check for the next ρ_J to flip a sign
            (u, K) = popnext!(T, dir)
            verbose ? println("Next u: $u") : nothing
            verbose ? println("Rate flip: $K") : nothing
        end

        #We evaluate the rate integral over the interval [t_prev, t]
        t_prev = t
        G_t_prev = G(t_prev, (flow_class, param, 0)) 
        
        t = evaluate_inverse(flow_class, u, param)
        
        verbose ? println("Previous time: $t_prev") : nothing
        verbose ? println("Time: $t") : nothing
        G_t = G(t, (flow_class, param, 0))
               
        verbose ? println("G-vector at time: $G_t") : nothing
        ΔG = G_t .- G_t_prev

        #We check if the integral has exceeded the threshold (and if so, it is terminal)
        if non_terminal 
            ΔΛ = dot(ΔG, F_positive)
            if ΔΛ ≥ remainder
                #The segment is terminal, but we use a break keyword so no need to flip the bool, technically.
                non_terminal = false

                #We have a root between t_prev and t -> t overshot:  
                #t_final can be found in [t_prev, t] and u_final in [u_prev, u]
                integration_parameters = (flow_class, param, remainder)
                u = find_zero(G, (u_prev, u), numerics.auxiliary_method, integration_parameters)

                #We compute the ΔG for the final time.
                t = evaluate_inverse(flow_class, u, param)
                G_t = G(t, (flow_class, param, 0))
                
                #We update the reverse rate integral. 
                ΔG = G_t .- G_t_prev
                Λ_rev -= dot(ΔG, F_negative)
                break
            else
                remainder -= ΔΛ
                Λ_rev -= dot(ΔG, F_negative)
            end
        else
            Λ_rev -= dot(ΔG, F_negative)
            break
        end

        #We need to adjust the F-vectors for the next segment
        if sign_vector[K] > 0
            @views F_positive -= M_adj_rho[K,:]
            @views F_negative += M_adj_rho[K,:]
        elseif sign_vector[K] < 0
            @views F_positive -= M_adj_rho[K,:]
            @views F_negative += M_adj_rho[K,:]
        else
            #sign_vector = 0 can technically happen, but should not (likely with prob 0). We throw an error for now.
            error("Zero sign vector! *Can* happen, but should not, generally.")
        end
        sign_vector[K] = -sign_vector[K]
    end

    #We update the velocity vector in its J:th component
    state.auxiliary[J] = u
    
    #Finally we adjust the forward rates. 
    u_tuple = (u^3, u^2, u, 1.0)
    fetch_rate_vector!(segment.forward_rates, evo_data, u_tuple, J, rate_is_fwd = true)

    return false
end

function fetch_rate_vector!(rate_vector, evo_data, u_tuple, J; rate_is_fwd::Bool)
    #The rate vector has N components, with the J:th one the probability to transition back to position space.
    #This is just applying ReLU
    #This has to be rearranged somewhat
    M_adj_rho = evo_data.split_data.M_adj_rhoJ
    sign_vector = evo_data.split_data.sign_vector
    for I in eachindex(rate_vector)
        if I ≠ J
            #We check if the fwd rate is positive by checking the sign vector
            if rate_is_fwd
                if sign_vector[I] > 0
                    @views rate_vector[I] = dot(M_adj_rho[I,:], u_tuple)
                else
                    rate_vector[I] = 0.
                end
            else
                if sign_vector[I] < 0
                    @views rate_vector[I] = -dot(M_adj_rho[I,:], u_tuple)
                else
                    rate_vector[I] = 0.
                end
            end
        else
            if rate_is_fwd
                if sign_vector[end] > 0
                    @views rate_vector[J] = dot(M_adj_rho[end,:], u_tuple)
                else
                    rate_vector[J] = 0.
                end
            else
                if sign_vector[end] < 0
                    @views rate_vector[J] = -dot(M_adj_rho[end,:], u_tuple)
                else
                    rate_vector[J] = 0.
                end
            end
        end
    end
end

function stationary_flow(threshold, F_positive, F_negative, sign_vector, ini_u_tuple)
    if !any(sign_vector .> 0)
        #If we had introduced a time in the velocities this wouldn't be a problem (though bad still)
        error("No positive rates - point must be rejected.")
    end
    ∂Λ = dot(F_positive, ini_u_tuple)
    if iszero(∂Λ)
        #This shouldn't happen as we should precisely consider those rates which give a non-negative integral.
        #The fringe case is, I suppose, the dot product being zero for all positive rates.
        #We could be more stringent in picking only the rates which are positive (rather than non-negative).
        #If the problem arises it would arise earlier in those cases.
        error("No positive forward rate, despite non-zero F-positive vector.")
    end
    t_final = threshold/∂Λ
    Λ_rev = -t_final*dot(F_negative, ini_u_tuple)
    return Λ_rev
end
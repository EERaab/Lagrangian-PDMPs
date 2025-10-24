function evaluate_flow!(pdmp::PDMP, segment::Segment{N}, state::SplitState, evo_data::EvolutionData, 
        numerics::NumericalParameters, dyn::VelocityODE; max_duration::Float64 = 0.0, reversed_pdmp::Bool = false) where N
        
        #We assume a single rate.
        if !(N == 1)
            error("Not implemented!")
        end

        #We pick a stopping threshold randomly.
        threshold = -log(rand())
        
        #The "parameters" (param) that determine our equation of motion and rates is partially encoded in evo_tensors.
        fetch_evo_data!(pdmp, evo_data, numerics, state, dyn)
        
        reset_integrator!(evo_data, state, threshold)

        #We explicitly assume that the events are of a single type here!
        #In a future implementation this could change.
        segment.reverse_rates[1] = -(integrator.f.f(long_trash_vector, u, t, integrator.p))[1]
        
        sol = solve!(evo_data.integrator)

        @views state.auxiliary .= sol.u[end][3:end]
        #We've encoded the overall rate integrals in the parameter-set of the integrator
        segment.forward_rate_integral = integrator.p[1][2] #should be ≈ threshold, up to float-acc
        segment.reverse_rate_integral = integrator.p[1][3]
        #Again, we assume there's only a single rate
        segment.forward_rates[1] = sol(sols.t[end], Val{1})[1]

        #We treat the volume adjustment factor as an adjustment of the reverse rate integral:
        if reversed_pdmp
            segment.reverse_rate_integral += sol_final[3]  
        else
            segment.reverse_rate_integral -= sol_final[3]
        end
        nothing
    end


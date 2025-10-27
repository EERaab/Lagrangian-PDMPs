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
        
        reset_integrator!(evo_data, state, threshold, reversed_pdmp)

        #We explicitly assume that the events are of a single type here!
        #In a future implementation this could change.
        integrator = evo_data.integrator
        segment.reverse_rates[1] = -(integrator.f.f(evo_data.long_trash_vector, evo_data.velocity_u0, integrator.p, 0.0))[1] #assumes that the rate is indeed positive here.
        sol = solve!(integrator)

        @views state.auxiliary .= sol.u[end][3:end]
        #We've encoded the overall rate integrals in the parameter-set of the integrator
        segment.forward_rate_integral = integrator.p[1][2] #should be ≈ threshold, up to float-acc
        segment.reverse_rate_integral = integrator.p[1][3]
        #Again, we assume there's only a single rate
        segment.forward_rates[1] = sol(sol.t[end], Val{1})[1]

        #We treat the volume adjustment factor as an adjustment of the reverse rate integral:
        if reversed_pdmp
            segment.reverse_rate_integral += sol.u[end][2]  
        else
            segment.reverse_rate_integral -= sol.u[end][2]
        end
        nothing
    end


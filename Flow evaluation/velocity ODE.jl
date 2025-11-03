function evaluate_flow!(threshold::Float64, pdmp::PDMP, segment::Segment{N}, state::SplitState, evo_data::EvolutionData, 
        numerics::NumericalParameters, dyn::VelocityODE; max_duration::Float64 = 0.0, reversed_pdmp::Bool = false) where N

        #The "parameters" (param) that determine our equation of motion and rates is partially encoded in evo_tensors.
        fetch_evo_data!(pdmp, evo_data, numerics, state, dyn)
        
        reset_integrator!(evo_data, state, threshold, reversed_pdmp)
        integrator = evo_data.integrator
        #We explicitly assume that the events are of a single type here!
        segment.reverse_rates[1] = (integrator.f.f(evo_data.long_trash_vector, evo_data.velocity_u0, integrator.p, 0.0))[2]
        #In a future implementation this could change.
        sol = solve!(integrator)


        segment.forward_rates[1] = get_du!(evo_data.long_trash_vector, integrator)[1]#(sol(sol.t[end], Val{1})[1]) #the interpolated derivative - alternatively we could use get_du!(evo_data.long_trash_vector, sol.u[end])[1]
        segment.forward_rate_integral = sol.u[end][1]
        segment.reverse_rate_integral = sol.u[end][2]

        @views state.auxiliary .= sol.u[end][4:end]

        #Note difference in definitions of ψ from old version -> current def will differ in sign from old version
        segment.reverse_rate_integral += sol.u[end][3]
        return false
    end


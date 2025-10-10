function evaluate_flow!(pdmp::PDMP, vertex::MethodVertex{VelocityODE}, state::SplitState, evo_data::EvolutionData, 
    numerics::NumericalParameters; max_duration::Float64 = 0.0, reversed_pdmp::Bool = false) 
    #We pick a stopping threshold randomly.
    threshold = -log(rand())
    
    #We will manipulate the segment indicated by the vertex.
    segment = pdmp.graph.segments[vertex.segment_number]

    #The "parameters" (param) that determine our equation of motion and rates.
    #Typically this may be quite messy to determine, and may require some more general function, but for the two methods (Lagrangian, BPS-Lagrangian) we consider it is sufficient to have
    fetch_evo_data!(pdmp, evo_data, numerics, state, dyn_type)
    
    #The functions we integrate follow the ODE du = dynamical!(du, u,...)dt
    dynamics_function!(du::MVector, u::MVector, evot, t) = velocity_dynamics!(du, u, pdmp, evot, dyn_type);
    
    #We set up the initial condition.
    #For BPS-Lagrangian and Lagrangian methods this is u0 = [[0.0, 0.0, 0.0]; state.auxiliary] but generally we may imagine a more advanced version.
    #Until a more advanced version is constructed the code below is acceptable
    u0 = MVector{3 + pdmp.target.dimension, Float64}([[0.0, 0.0, 0.0]; state.auxiliary])
    du = zeros(MVector{3 + pdmp.target.dimension, Float64})

    #We must compute the initial reverse rates.
    #Generally this could be more advanced, but for Lagrangian and BPS-Lagrangian methods there is only a single rate to set.
    #Until a more advanced version is constructed the code below is acceptable
    dynamics_function!(du, u0, evo_data.evo_tensors, 0.0)    
    segment.reverse_rates[1] = du[2]

    #We fix the ODE problem
    problem = ODEProblem(dynamics_function!, u0, 10000.0, evo_data.evo_tensors)        
    integrator = init(problem, numerics.auxiliary_method)
    condition(u, t, integrator) = (u[1] - threshold)  
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    counter = 0
    while true
        sol = solve(problem, callback = cb, save_on=false, save_start=false, save_end=true)

        #If we reached the point at which the forward rate intgral reaches the threshold we terminate.
        if sol.retcode == ReturnCode.Terminated
            #The final state in the solver is our output
            sol_final = sol.u[end]

            #The velocity is updated
            state.auxiliary .= @view sol_final[4:end]

            #The rate integrals are updated
            segment.forward_rate_integral = sol_final[1]
            segment.reverse_rate_integral = sol_final[2]
            
            #The forward rate is added.
            #NOTE: This may be zero is some cases, which will lead to NaN-type answers.
            #The appropriate forward rate is the final (before termination) non-zero approximated rate,
            #i.e. the interpolated fwd - rate.
            #In practice if it is infinitesimally small the acceptance rate is very large 
            segment.forward_rates[1] = sol(sol.t[end], Val{1})[1]

            #Depending on the PDMP direction we pick one volume adjustment or the other. 
            #Rather than keeping this as its own factor in the segment, we map it to the reversed rate integral.
            if reversed_pdmp
                segment.reverse_rate_integral += sol_final[3]  
            else
                segment.reverse_rate_integral -= sol_final[3]
            end

            #We have updated the state and segment, and view the transformation as instant (Δt = 0.0) but dynamical so we return
            return 0.0
        end

        #If we did not terminate we set the new initial state to the final state of our integral and start over.
        u0 .= sol.u[end]

        #To avoid getting stuck in infinite loops when there are bugs or instabilities we add termination conditions.
        if any(abs.(u0[4:end]) .> 10^6)
            @show u0
            @show sol(sol.t[end],Val{1})[1]
            @show counter
            error("Rho is bebeyond 10^6. Stopping.")
        end
        counter +=1
        if counter > 100
            @show u0
            @show du
            @show counter
            error("Looped too many times.")
        end
    end
end
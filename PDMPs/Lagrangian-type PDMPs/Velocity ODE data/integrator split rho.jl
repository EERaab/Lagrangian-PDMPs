include("adjusted reinit.jl")

#### INTEGRATOR HANDLING ####

    function reset_integrator!(evo_data, state, stoch_threhold, reversed_pdmp; adjusted = true)
        integrator = evo_data.velocity_ode_data.integrator
        #The parameters of our integrator are a tuple integrator.p = (data_vec, EvoTensor, reversed_pdmp)
        #We must reset the first element, an MVector of length 1.
        data_vec = integrator.p[1] # = (threshold)
        data_vec[1] = stoch_threhold

        #Our new initial rate/velocity is ∼[0,0,0, state.velocity]
        u0 = evo_data.velocity_ode_data.velocity_u0
        u0[1] = 0.0 #Corresponds to ∫λ_fwd dt (wrt Z which may or may not be reversed)
        u0[2] = 0.0 #Corresponds to ∫λ_rev dt (wrt Z which may or may not be reversed)
        u0[3] = 0.0 #Our ψ = ∫ div(Φ_v)dt = ∫ tr(Γ) ⋅ vdt
        @view(u0[4:end]) .= state.auxiliary #the velocity.

        adjusted_reinit!(integrator, u0, run_adjusted = adjusted)
        #reinit!(integrator, evo_data.velocity_u0)

        integrator.p[3].x = reversed_pdmp
        nothing
    end

    function initialize_integrator(param, nums::NumericalParameters, u0)
        #We terminate when the (positive part of the) du[1] integral exceeds some certain value
        threshold_condition(u, t, integrator) = integrator.p[1][1] - u[1]
        threshold_affect!(integrator) = terminate!(integrator)

        total_cb = ContinuousCallback(threshold_condition, threshold_affect!, save_positions = (true, true))

        dynamics_function!(du, u, p, t) = velocity_dynamics_split_rho!(du, u, p)#velocity_dynamics!(du, u, p, pdmp, trash_vector)
        jacobian_function!(J, u , p, t) = jacobian_dynamics_split_rho!(J, u, p)#jacobian_dynamics!(J, u, p, pdmp)

        ff = ODEFunction(dynamics_function!, jac = jacobian_function!)
        problem = ODEProblem(ff, u0, Inf, param)
        solver = nums.auxiliary_method 
        integrator = init(problem, solver, callback = total_cb)
        return integrator
    end

#### THE END ####
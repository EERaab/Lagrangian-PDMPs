include("adjusted reinit.jl")

#### INTEGRATOR HANDLING ####

    function reset_integrator!(evo_data, state, stoch_threhold)
        integrator = evo_data.integrator
        #The parameters of our integrator are a tuple integrator.p = (data_vec, EvoTensor)
        #We must reset the first element, an MVector of length 3.
        data_vec = integrator.p[1]
        data_vec[1] = stoch_threhold
        data_vec[2] = data_vec[3] = 0.0

        #Our new initial rate/velocity is ∼[0,0, state.velocity]
        u0 = evo_data.velocity_u0
        u0[1] = 0.0 #Corresponds to ∫ρ dt
        u0[2] = 0.0 #Our ψ = ∫ div(Φ_v)dt = ∫ tr(Γ) ⋅ vdt
        @view(u0[3:end]) .= state.auxiliary #the velocity.

        adjusted_reinit!(integrator, evo_data.velocity_u0, run_adjusted = true)
        nothing
    end

    function downcrossing_affect!(integrator)
        #We take the 2nd parameter (our fwd rate integral) and add to it the positive part of the rho integral
        integrator.p[1][2] = integrator.u[1]
        #We reset the rho integral to 0.0
        #(could cause issues if u[1] = 0 when du[1] = 0)
        #(Try adjusting down the threshold instead if issues arise)
        integrator.u[1] = 0.0
        nothing
    end

    function upcrossing_affect!(integrator)
        #We take the third parameter (our reverse rate integral) and add the negative of the rho integral
        integrator.p[1][3] -= integrator.u[1]
        #We adjust our rho integral to be equal to the value of the previous positive parts
        integrator.u[1] = integrator.p[1][2]
        nothing
    end

    function initialize_integrator(pdmp::PDMP, param, nums::NumericalParameters, long_trash_vector, trash_vector, trash_matrix, u0)
        #We terminate when the (positive part of the) du[1] integral exceeds some certain value
        threshold_condition(u, t, integrator) = integrator.p[1][1] - u[1]
        threshold_affect!(integrator) = terminate!(integrator)

        termination_cb = ContinuousCallback(threshold_condition, threshold_affect!, save_positions = (false, true))

        #We cross when du[1] = 0.0
        #This can be directly implemented as M or taken as M = integrator.f.f(u)[1] 
        crossing_condition(u, t, integrator) = (integrator.f.f(long_trash_vector, u, integrator.p, t))[1] #du[1] = velocity_dynamics!(...)[1]

        crossing_cb = ContinuousCallback(crossing_condition, upcrossing_affect!, affect_neg! = downcrossing_affect!, save_positions = (true, true))
        total_cb = CallbackSet(termination_cb, crossing_cb)

        dynamics_function!(du, u, p, t) = velocity_dynamics!(du, u, p, pdmp, trash_vector)#velocity_dynamics!(du, u, p, pdmp, trash_vector)
        jacobian_function!(J, u , p, t) = jacobian_dynamics!(J, u, p, pdmp, trash_matrix, trash_vector)#jacobian_dynamics!(J, u, p, pdmp)

        ff = ODEFunction(dynamics_function!; jac = jacobian_function!)
        problem = ODEProblem(ff, u0, Inf, param)
        solver = nums.auxiliary_method #AutoTsit5(Rosenbrock23())
        integrator = init(problem, solver, callback = total_cb, save_everystep=false)
        return integrator
    end

#### THE END ####
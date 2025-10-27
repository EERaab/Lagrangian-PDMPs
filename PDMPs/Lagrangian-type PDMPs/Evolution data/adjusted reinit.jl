##################### DEFINES RUNS AN ADJUSTED VERSION OF 'reinit!' ################################
#By specializing to our case where we reuse the same tstops, d_discontinuities etc over and over
#we can avoid allocations in re-initialization of the integrator.
#
#Functions exactly like 'reinit!' of OrdinaryDiffEq fame, except inside indicated blocks.


function adjusted_reinit!(integrator::OrdinaryDiffEqCore.ODEIntegrator, u0 = integrator.sol.prob.u0;
        t0 = integrator.sol.prob.tspan[1],
        tf = integrator.sol.prob.tspan[2],
        erase_sol = true,
        tstops = integrator.opts.tstops_cache,
        saveat = integrator.opts.saveat_cache,
        d_discontinuities = integrator.opts.d_discontinuities_cache,
        reset_dt = (integrator.dtcache == zero(integrator.dt)) &&
                   integrator.opts.adaptive,
        reinit_dae = true,
        reinit_callbacks = true, initialize_save = true,
        reinit_cache = true,
        reinit_retcode = true,
        run_adjusted = false)
    if reinit_dae && SciMLBase.has_initializeprob(integrator.sol.prob.f)
        # This is `remake` infrastructure. `reinit!` is somewhat like `remake` for
        # integrators, so we reuse some of the same pieces. If we pass `integrator.p`
        # for `p`, it means we don't want to change it. If we pass `missing`, this
        # function may (correctly) assume `newp` aliases `prob.p` and copy it, which we
        # want to avoid. So we pass an empty array of pairs to make it think this is
        # a symbolic `remake` and it can modify `newp` inplace. The array of pairs is a
        # const global to avoid allocating every time this function is called.
        u0,
        newp = SciMLBase.late_binding_update_u0_p(integrator.sol.prob, u0,
            EMPTY_ARRAY_OF_PAIRS, t0, u0, integrator.p)
        if newp !== integrator.p
            integrator.p = newp
            sol = integrator.sol
            @reset sol.prob.p = newp
            integrator.sol = sol
        end
    end

    if isinplace(integrator.sol.prob)
        recursivecopy!(integrator.u, u0)
        recursivecopy!(integrator.uprev, integrator.u)
    else
        integrator.u = u0
        integrator.uprev = integrator.u
    end

    if OrdinaryDiffEqCore.alg_extrapolates(integrator.alg)
        if isinplace(integrator.sol.prob)
            recursivecopy!(integrator.uprev2, integrator.uprev)
        else
            integrator.uprev2 = integrator.uprev
        end
    end



    integrator.t = t0
    integrator.tprev = t0


    tType = typeof(integrator.t)
    ######################### THIS BLOCK OF CODE ALLOCATES#######################################
    #Fix: ??? 
    #Have to deal with tstops, saveat and d_discontinuities.
    #Since saveat, d_discontinuities and tstops are all the same, we can keep them as
    #they are. Have to avoid adjustments though to avoid bugs here.

    if run_adjusted
        #This should be used with care. We use a check here to ensure that we empty the t-stops if the state of the integrator is not as expectted.
        while !isempty(integrator.opts.tstops)
            #print("Non-empty t-stop heap.")
            pop!(integrator.opts.tstops)
        end
        push!(integrator.opts.tstops, tType(tf))
    else
        tspan = (tType(t0), tType(tf))
        integrator.opts.tstops = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, tspan)
        integrator.opts.saveat = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, tspan)
        integrator.opts.d_discontinuities = OrdinaryDiffEqCore.initialize_d_discontinuities(tType,
            d_discontinuities,
            tspan)
    end
    ######################### END OF ALLOCATING BLOCK #######################################


    if erase_sol
        if integrator.opts.save_start
            resize_start = 1
        else
            resize_start = 0
        end
        resize!(integrator.sol.u, resize_start)
        resize!(integrator.sol.t, resize_start)
        resize!(integrator.sol.k, resize_start)

        if integrator.opts.save_start || (!isempty(saveat) && saveat[1] == tType(t0))
            copyat_or_push!(integrator.sol.t, 1, t0)
            if integrator.opts.save_idxs === nothing
                copyat_or_push!(integrator.sol.u, 1, u0)
            else
                u_initial = u0[integrator.opts.save_idxs]
                copyat_or_push!(integrator.sol.u, 1, u_initial, Val{false})
            end
        end
        if integrator.sol.u_analytic !== nothing
            resize!(integrator.sol.u_analytic, 0)
        end
        if integrator.alg isa OrdinaryDiffEqCore.OrdinaryDiffEqCompositeAlgorithm
            resize!(integrator.sol.alg_choice, resize_start)
        end
        integrator.saveiter = resize_start
        if integrator.opts.dense
            integrator.saveiter_dense = resize_start
        end
    end
    integrator.iter = 0
    integrator.success_iter = 0
    integrator.u_modified = false

    # full re-initialize the PI in timestepping
    OrdinaryDiffEq.reinit!(integrator, integrator.opts.controller)
    integrator.qold = integrator.opts.qoldinit
    integrator.q11 = typeof(integrator.q11)(1)
    integrator.erracc = typeof(integrator.erracc)(1)
    integrator.dtacc = typeof(integrator.dtacc)(1)

    if reset_dt
        auto_dt_reset!(integrator)
    end

    if reinit_dae &&
       (integrator.isdae || SciMLBase.has_initializeprob(integrator.sol.prob.f))
        DiffEqBase.initialize_dae!(integrator)
        update_uprev!(integrator)
    end

    if reinit_callbacks
        OrdinaryDiffEqCore.initialize_callbacks!(integrator, initialize_save)
    end

    if reinit_cache
        OrdinaryDiffEqCore.initialize!(integrator, integrator.cache)
    end

    if reinit_retcode
        integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, ReturnCode.Default)
    end
    return nothing
end
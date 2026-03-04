using LinearAlgebra, BenchmarkTools, PolynomialRoots, StaticArrays, DataStructures, Statistics, Roots

# We collect the velocity functions in a data structure
    struct VelocityFunctions{T,F,G}
        velocity::T
        inverse::F
        integral::G 
    end

    function evaluate(V::VelocityFunctions, t, param)
        return V.velocity(t, param...)
    end

    function evaluate_inverse(V::VelocityFunctions, u, param)
        return V.inverse(u, param...)
    end

    function evaluate_integral(V::VelocityFunctions, t, param)::NTuple{4, Float64}
        return V.integral(t, param...)
    end

####

#Type 0 
    #Parameters: (u0)
    function type_0(t, u0)
        return u0
    end

    #type_0 obviously has no inverse, and we treat its integral as a special case.
    type0 = VelocityFunctions(type_0, :None, :None)

####

#Type 1
    #Parameter tuple: c, u0, inv_c
    function type_1(t, c, u0, inv_c)
        return c*t + u0
    end

    function type_1_inv(u, c, u0, inv_c)
        return (u-u0)*inv_c
    end

    function G_type_1(T, c, u0, inv_c)
        return (T^4, T^3, T^2, T) ./ (4.,3.,2.,1.)
    end
    type1 = VelocityFunctions(type_1, type_1_inv, G_type_1)

####

#Type 2
    #Param b, q, m
    function type_2(t, b, q, m)
        return m*exp(b*t)-q
    end

    function type_2_inv(u, b,q, m)
        return log((u+q)/m)/b
    end

    #Param: b, q, m
    function G_type_2(T, b, q, m)
        s = exp(b*T)
        return ((s^3)/3.0,(s^2)/2.0,s,T) ./ b
    end
    type2 = VelocityFunctions(type_2, type_2_inv, G_type_2)

####

#Type 3
    #Parameters (a_inv, l0_inv, δ, a)
    function type_3(t, a_inv, l0_inv, δ, a)
        #a_inv = 1/a
        #l0_inv = 1/(a*u_0 + δ)
        return a_inv*((1/(l0_inv-t)) - δ)
    end

    function type_3_inv(u, a_inv, l0_inv, δ, a)
        return l0_inv - (1/(a*u + δ))
    end

    #Param: (a_inv, l0_inv, l0, δ, a)
    function G_type_3(T, a_inv, l0_inv, l0, δ, a)
        l0 = l0
        Δ = l0 - T
        Δ_inv = inv(Δ)
        return ((Δ_inv^2)/2.0, Δ_inv, -log(Δ), T) 
    end

    type3 = VelocityFunctions(type_3, type_3_inv, G_type_3)

####

#Type 4
    #Parameters (a_inv, k, s0, δ, k_inv, a)
    function type_4(t, a_inv, k, s0, δ, k_inv, a)
        return a_inv*((k * tan(s0 + (k*t))) - δ)
    end

    function type_4_inv(u, a_inv, k, s0, δ, k_inv, a)
        return k_inv*(-s0 + atan(k_inv*((a*u)+δ)))
    end

    #Parameters (a_inv, k, s0, δ, k_inv, a)
    function G_type_4(T, a_inv, k, s0, δ, k_inv, a)
        X = (k*T) + s0
        sX, cX  = sincos(X)
        lcX = log(cX^2)
        tX = sX/cX
        return ( tX^2 + lcX, tX-X, lcX, T) .* (k_inv/2.0, k_inv, -k_inv/2.0, 1.0)
    end

    type4 = VelocityFunctions(type_4, type_4_inv, G_type_4)

####

#Type 5
    #Parameters: (a_inv, δk, dk, ω0, a)
    function type_5(t, a_inv, δk, dk, ω0, a)
        return a_inv*(-δk + (dk/(1 - ω0*exp(dk*t))))
    end

    function type_5_inv(u, a_inv, δk, dk, ω0, a)
        return log((1-(dk/((a*u) + δk)))/ω0)/dk
    end

    function G_type_5(T, a_inv, δk, dk, ω0, a)
        X = dk*T 
        Y = ω0*exp(X)-1.
        Z = abs(dk*Y/2.0)
        first_term = X-log(Z)
        exp_inverse= inv(Y)
        second_term = dk * ( first_term - exp_inverse)
        third_term = (dk^2) * (first_term - ((Y - 0.5) *(exp_inverse^2)))
        return (third_term, second_term, first_term, T)
    end

    type5 = VelocityFunctions(type_5, type_5_inv, G_type_5)

####

# Computing direction, velocity type 
    #We want a method for the sign that returns an integer no matter what (unlike the default sign function in Julia, which returns the sign with corresponding type)
    function sgn(x)::Int64
        if x < 0
            return -1
        elseif x > 0
            return 1
        end
        return 0
    end

    # Direction and velocity
    function direction_velocity_and_type(a,b,c,u0)
        if isapprox(a*(u0^2) + (b*u0) , -c)
            @warn "Stationary velocity, rates are constant"
            return (0,  (u0,), type0)
        end
        if a ≈ 0 
            if b ≈ 0 
                if c ≈ 0 #Techincally shouldn't happen, but floats are floats
                    @warn "Stationary velocity, rates are constant"
                    return (0,  (u0,), type0)
                end
                return return (sgn(c), (c, u0, inv(c)), type1)
            end
            q = c/b
            return (sgn(u0*b + c), (b, q, u0+b), type2)
        end
        #β = b
        #γ = c*a
        #y0 = a*u0
        δ = b/2
        l0 = a*u0 + δ
        X1 = 4*c*a
        X2 = b^2
        a_inv = inv(a)
        if isapprox(X1, X2)
            l0_inv = inv(l0)         
            return (sgn(a), (a_inv, l0_inv, δ, a), type3)
        end 
        κ2 = (X1-X2)
        k = sqrt(abs(κ2))/2
        if κ2 > 0
            s0 = atan(l0/k)# t0 = s0/k
            return (sgn(a), (a_inv, k, s0, δ, inv(k), a), type4)

        elseif κ2 <0 #should be an 'else', but we'd rather get errors than bugs
            dk = 2*k
            ω_0 = 1 - (dk/(l0 + k)) 
            δk = δ + k
            return (sgn(ω_0*a), (a_inv, δk, dk, ω_0, a), type5)
        end
        error("Comparison failure.")
    end

####
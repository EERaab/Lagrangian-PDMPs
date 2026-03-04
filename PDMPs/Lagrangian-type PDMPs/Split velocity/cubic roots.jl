# A naive method for cubic polynomial evalutation
    function eval_cubic(b,c,d,x)
        #Turns out to be fast enough. I <3 compiler
        return x^3 + b*x^2 + c*x + d
    end

####

# A function to check for 'upcoming' sign flips in cubics 
# I.e. sign flips of a cubic that occurs for x > x0 
    function upcoming_sign_flips!(T, b, c, d, x0, dir; indicator)
        A1 = b^2
        A2 = 3*c

        #We do a few simple checks to avoid computations when roots are not present in the set.
        if eval_cubic(b,c,d,x0) > 0
            if A1 > A2 #local minima exists
                x_min = (sqrt(A1 - A2) - b)/3
                if !(x_min > x0)
                    return T
                end
                if !(eval_cubic(b,c,d,x_min) < 0)
                    return T
                end
                #If neither of the above are fulfilled we know that we shall pass the minimum 
                #which is negative in function value, so at least 2 more roots are to come!
            else
                #No local minima exist and x^3+... is monotonically increasing. Hence no roots
                return T
            end
        end

        #Triple root
        if (A1 ≈ A2) && ((2*(b^3)) + (27*d) ≈ (9*b*c)) 
            root = -b/3
            if root > x0
                push!(T, (dir*root,indicator))
            end
            return T 
        end

        Δ = A1-A2
        μ = (2*(b^3)) - (9*b*c) + (27*d)
        mu_squared = (μ^2)
        four_delta = (4*Δ^3)
        L = mu_squared - four_delta
        if mu_squared < four_delta
            z = (μ + sqrt(-L)*im) #2z = ...,  but we only use the angle for z
            r2 = cbrt((μ^2 - L)/4)
            if r2 ≈ Δ 
                (s, c) = sincos(angle(z)/3) #can be optimized/altered to use the 'tan-formula' for the 3xReal root cubic
                k = 2. * sqrt(r2)
                roots = (b .+ (k.* (c, (-c + (sqrt(3.)*s))/2., (-c - (sqrt(3.)*s))/2.))) ./(-3.) #awful
                for root in roots
                    if root > x0
                        push!(T, (dir*root,indicator))
                    end
                end
                return T
            else
                if μ ≤ 0
                    root = (b + sqrt(r2)*(1+(Δ/r2))) ./(-3)
                else
                    root = (b - sqrt(r2)*(1+(Δ/r2)))./(-3)
                end
                if root > x0
                    push!(T, (dir*root,indicator))
                end
                return T
            end
        elseif mu_squared ≥ four_delta #L ≥ 0
            Lroot = sqrt(L)
            if μ > 0
                z = (μ + Lroot)/2.0
            else
                z = (μ - Lroot)/2.0
            end
                
            C = cbrt(z)
            r2 = C^2
            if r2 ≈ Δ 
                roots = (b+2*C, b-C) ./ (-3)
                for root in roots
                    if root > x0
                        push!(T, (dir*root,indicator))
                    end
                end
                return T
            else
                root =  (b + C*(1+(Δ/r2))) /(-3)
                if root > x0
                    push!(T, (dir*root,indicator))
                end
                return T
            end
        else
            error("Some comparison bug")
        end
    end

####

# A function that checks for sign flips for sub-cubic polynomials for x > x0
    function sub_cubic_sign_flips!(T, b, c,  d, x0, dir; indicator)
        if b ≈ 0 #not very pleasing
            if c ≈ 0 #does not spark joy
                return T
            end
            root = (-d/c)
            if root > x0
                push!(T, (dir*root,indicator)) 
            end
            return T
        end
        β = c/b
        γ = d/b
        A1 = β^2
        A2 = 4*γ
        if !(A1 > A2)
            return T
        end
        Δ = sqrt(A1-A2)
        root1 = (-β - Δ)/2
        root2 = (-β + Δ)/2
        if root1 > x0
            push!(T, (dir*root1,indicator))
            push!(T, (dir*root2,indicator))
        elseif root2 > x0
            push!(T, (dir*root2,indicator))
        end
        return T
    end

####

# A function that checks for sign flips for at most cubic polynomials.
# Checks for x>x0, or (if dir = -1), reverses the polynomial p(x) to check for p(-x):s sign flips.
    function upcoming_sign_flips!(T, a, b, c, d, x0, dir; indicator)
        if a ≈ 0 #not very pleasing
            if dir > 0
                return sub_cubic_sign_flips!(T, b, c, d, x0, dir, indicator = indicator )
            else
                #we map x -> y = -x to avoid having to work 'both' cases
                return sub_cubic_sign_flips!(T, b, -c, d, -x0, dir, indicator = indicator )
            end
        end
        if dir > 0
            return upcoming_sign_flips!(T, b/a, c/a, d/a, x0, dir, indicator = indicator)
        else
            #we map x -> y = -x to avoid having to work 'both' cases
            return upcoming_sign_flips!(T, -b/a, c/a, -d/a, -x0, dir, indicator = indicator)
        end
    end

####

# A function that creates a partition of (u0, ∞) (or  (-∞, u0) if dir = -1) by dividing when any
# cubic polynomial (coefficients corresponding to a row vector of ρJ) changes its sign.
    function partition_by_roots!(T::BinaryMinHeap, J::Integer, ρJ::Array{Float64, 2}, dir::Integer, u0::Float64) 
        empty!(T)
        if dir == 0 
            return T
        end

        for I in axes(ρJ, 1)
            if I ≠ J
                ρJI = @view(ρJ[I,:])
                upcoming_sign_flips!(T, ρJI[1], ρJI[2], ρJI[3], ρJI[4], u0, dir, indicator = I)
            end
        end
        return T
    end
    
    function partition_by_roots!(J::Integer, evo_data::SplitEvoData, dir::Integer, u0::Float64) 
        return partition_by_roots!(evo_data.velocity_partition, J, evo_data.signed_rho_J, dir, u0)
    end
####

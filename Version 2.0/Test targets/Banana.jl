"""
    log_banana_density(x::Vector{<:Real}; b=0.1, c=1.0)

Computes the log-density of a 2D banana-shaped distribution.

# Arguments
- `x`: a 2-element vector [x₁, x₂]
- `b`: banana bend parameter (default 0.1)
- `c`: expected value of x₁² under the base Gaussian (default 1.0)

# Returns
- log-density value (Float64)
"""
function log_banana_density(x::Vector{<:Real}, sig = 1., B = 1., mu = 1.)
    return -x[1]^2/2/sig-(x[2]+B*x[1]^2-mu)^2/2
    #x1, x2 = x[1], x[2]
    # Transform: x2 becomes banana-shaped
    #y1 = x1
    #y2 = x2 - b * (x1^2 + c)

    # Standard bivariate normal log-density
    #return -0.5 * (y1^2 + y2^2) - log(2π)
end


target = TargetData(log_banana_density, 2)


function log_2drosenbrock_density(x::Vector{<:Real})
    a = 1/20 
    b = 1000*100/20
    return - b * (x[2] - x[1]^2)^2 - a*(1-x[1])^2
end

target_rosenbrock = TargetData(log_2drosenbrock_density,2)

rosenbrock_CDF = x -> cdf(Normal(1,sqrt(1/(2*1/20))),x)
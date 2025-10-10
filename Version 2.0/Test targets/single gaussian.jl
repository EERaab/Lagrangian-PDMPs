function log_single_gaussian_2D(X::Vector)
    sum1 = sum(X.^2)
    return (-sum1/2.0) - log(2*π)/2
end

function normal(μ::Float64, C::Float64)
    return X -> (sqrt(2*π)^length(μ)*sqrt(C))^(-1)*exp(-((X-μ)^2)/(2*C))
end

x_marginal_distr(x) = (normal(0.0, 1.0)(x))
y_marginal_distr(y) = (normal(0.0, 1.0)(y))

target = TargetData(log_single_gaussian_2D, 2)


#Below we define the analytical derivatives of the single gaussian. Not necessary if we want to run ForwardDiff.
function log_single_gaussian_2D_gradient!(grad::Vector{Float64}, position::Vector{Float64})
    for i in eachindex(grad)
        @inbounds grad[i] = -position[i]
    end
    return grad
end

function log_single_gaussian_2D_hessian!(hess::Matrix{Float64}, position::Vector{Float64})
    for i in eachindex(position), j in eachindex(position)
        if !(i == j) 
            @inbounds hess[i,j] = 0.0
        else
            @inbounds hess[i,j] = -1.0
        end
    end
    return hess
end

function log_single_gaussian_2D_full!(third::Array{Float64,3}, position::Vector{Float64})
    for i ∈ eachindex(third)
        third[i] = 0.0
    end
    return third
end

function log_single_gaussian_2D_directional!(dirhess::Matrix{Float64}, position::Vector{Float64}, velocity::Vector{Float64})
    for i ∈ eachindex(dirhess)
        dirhess[i] = 0.0
    end
end

analytical_diffs = AnalyticalDer(log_single_gaussian_2D_gradient!, log_single_gaussian_2D_hessian!, log_single_gaussian_2D_directional!, log_single_gaussian_2D_full!)
function log_twin_peaks_20D(X::Vector)
    sum1 = dot(X,X)
    sum2 = sum1 - sum(X) + 5.0#dot((X .-0.5),(X .-0.5))
    return log(exp(-sum1/2.0)+exp(-sum2/2.0)) - 10*log(2*π)
end

function normal(μ::Float64, C::Float64)
    return X -> (sqrt(2*π)^length(μ)*sqrt(C))^(-1)*exp(-((X-μ)^2)/(2*C))
end

x_marginal_distr(x) = (normal(0.0, 1.0)(x)+normal(0.5, 1.0)(x))/2
y_marginal_distr(y) = (normal(0.0, 1.0)(y)+normal(0.5, 1.0)(y))/2

function grad_tp_20D!(grad::Vector{Float64}, X::Vector{Float64})::Vector{Float64}
    shift = (0.5/(1+exp((-sum(X)/2)+2.5))) 
    @inbounds for i in eachindex(grad)
        grad[i] = -X[i] + shift
    end
    return grad
end


function hess_tp_20D!(hess::Matrix{Float64}, X::Vector{Float64})
    R = exp((-sum(X)/2)+2.5)
    fill!(hess, R*0.25/(1+R)^2)
    for i in eachindex(X)
        @inbounds hess[i,i] += -1        
    end
    return hess
end


function jachess_tp_20D!(jachess::Array{Float64,3}, X::Vector{Float64})
    R = exp((-sum(X)/2)+2.5)
    fill!(jachess, 0.125*(R-R^2)/(R+1)^3)
    return jachess
end
    

function dirhess_tp_20D!(dirhess::Matrix{Float64}, X::Vector{Float64}, V::Vector{Float64})
    R = exp((-sum(X)/2)+2.5)
    sv = sum(V)
    fill!(dirhess, 0.125*sv*(R-R^2)/(R+1)^3)
    return dirhess
end

ders = LagrangianDerivatives(grad_tp_20D!, hess_tp_20D!, jachess_tp_20D!, dirhess_tp_20D!, false)

target = TargetData(log_twin_peaks_20D, 20)
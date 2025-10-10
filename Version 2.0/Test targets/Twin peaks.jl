using Distributions

function twin_peaks(X::Vector)
    sum1= 10.0*(X[1]^2)+(5.0/3)*(X[2]^2)
    sum2 = (5.0/3)*((X[1]-1.5)^2) + 25.0*(X[2]^2)
    return log(((exp(-sum1/2.0)/(2*π*sqrt(0.06)))+(exp(-sum2/2.0)/(2*π*sqrt(0.024))))/2)
end

#To plot marginals we include a def of normals:
function normal(μ::Float64, C)
    return X -> (sqrt(2*π)^length(μ)*sqrt(C))^(-1)*exp(-((X-μ)^2)/(2*C))
end

#The distribution above is actually a normal of the form
μ1 = [0.0, 0.0]
μ2 = [1.5, 0.0]
C1 = 0.4*Diagonal([0.25, 1.5 ])
C2 = 0.4*Diagonal([1.5, 0.1])
alt_form(X) = log((normal(μ1,C1)(X) + normal(μ2,C2)(X))/2)

x_marginal_distr(x) = (normal(μ1[1],(C1[1,1]))(x)+normal(μ2[1],(C2[1,1]))(x))/2
y_marginal_distr(x) = (normal(μ1[2],(C1[2,2]))(x)+normal(μ2[2],(C2[2,2]))(x))/2

x_CDF = x -> (cdf(Normal(μ1[1],sqrt(C1[1,1])), x) + cdf(Normal(μ2[1],sqrt(C2[1,1])), x))/2.

target = TargetData(twin_peaks, 2)
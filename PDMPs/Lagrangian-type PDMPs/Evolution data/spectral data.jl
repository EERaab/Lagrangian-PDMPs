"""
    initialize_spectral_data(dimension::Int)::SpectralData

Returns a full zero-state of SpectralData, of the appropriate dimension. 
"""
function initialize_spectral_data(dimension::Int)::SpectralData
    return SpectralData(
        zeros(Float64, dimension, dimension), 
        zeros(Float64, dimension, dimension), 
        Diagonal(zeros(Float64,dimension)),
        HermitianEigenWs(zeros(Float64, dimension, dimension), vecs = true)
        )
end


"""
    jmatrixfunction1(λ::Float64, a::Float64)

Return the value λcoth(λa), or, if λ=0, the appropriate continuous extension, 1/a.
"""
function jmatrixfunction1(λ::Float64, a::Float64)
    if !iszero(λ)
        return λ*coth(λ*a)
    end
    return 1/a
end

"""
    jmatrixfunction2(λ::Float64,a::Float64)

Returns the value of the derivative (∂f/∂λ) at (λ,a) for f(λ,a)=λcoth(λa), or, if λ=0 the cont. extension, which happens to be zero.
"""
function jmatrixfunction2(λ::Float64,a::Float64)
    if iszero(λ)
        return 0.0
    end
    return coth(λ*a)-a*λ*(csch(a*λ))^2
end

"""
    fetch_spectral_data!(spec::SpectralData, hessian::Matrix{Float64}, hardness::Float64)

Takes the eigendecomposition specM of a matrix M and updates the spectral data in spec to match specM, given the hardness.
"""
function fetch_spectral_data!(spec::SpectralData, hessian::Matrix{Float64}, hardness::Float64)
    LAPACK.syevr!(spec.ws, 'V', 'A', 'U', hessian, 0.0, 0.0, 0, 0, -1.0)
    spec.Q .= spec.ws.Z
    λs = spec.ws.w
    spec.Dinv.diag .= 1 ./ jmatrixfunction1.(λs, hardness)
    for i ∈ eachindex(λs)
        for j ∈ eachindex(λs)
            if λs[i] == λs[j]
                @inbounds spec.jmatrix[i,j] = jmatrixfunction2(λs[i], hardness)
            else
                @inbounds spec.jmatrix[i,j] = (jmatrixfunction1(λs[i], hardness) - jmatrixfunction1(λs[j], hardness))/(λs[i] - λs[j])
            end
        end
    end
    return spec
end
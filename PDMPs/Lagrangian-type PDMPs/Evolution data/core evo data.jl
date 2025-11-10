
#While the numerical parameters specify "how" we compute derivatives and etc, the specific values that we compute are stored in EvolutionData structures.
#Because of the complexity in the Lagrangian methods we shall need several "sub-structures".

#Conveniently all Lagrangian-type methods share the same basic evolution data.

    #The pointwise numerical derivatives are stored in PointData.
    mutable struct PointData{DRP, DRV}
        gradient::Vector{Float64}
        position_update_data::DRP
        velocity_update_data::DRV

        function PointData(dim)
            gr = zeros(Float64, dim)
            pud = DiffResults.DiffResult(zeros(Float64, dim, dim), zeros(Float64, dim, dim))
            vud = DiffResults.DiffResult(zeros(Float64, dim, dim), zeros(Float64, dim, dim, dim))
            new{typeof(pud), typeof(vud)}(gr, pud, vud)
        end
    end

    #Given the pointwise data we also store spectral data about the hessian.
    struct SpectralData
        Q::Matrix{Float64}
        jmatrix::Symmetric{Float64, Matrix{Float64}}
        Dinv::Diagonal{Float64, Vector{Float64}}
        ws::HermitianEigenWs{Float64, Matrix{Float64}, Float64}

        function SpectralData(dim)
            Q = zeros(Float64, dim, dim)
            J = Symmetric(zeros(Float64, dim, dim), :U)
            Dinv = Diagonal(zeros(Float64, dim))
            ws = HermitianEigenWs(zeros(Float64, dim, dim), vecs = true)
            new(Q, J, Dinv, ws)
        end
    end 
        
    include("point data.jl")
    include("spectral data.jl")
    include("evo tensors.jl")

#The data used in all Lagrangian methods.
struct LagrangianCoreData{PD, T}
    point_data::PD
    spectral_data::SpectralData
    evo_tensors::T

    function LagrangianCoreData(dim)
        ini_pd = PointData(dim)
        ini_sd = SpectralData(dim)
        ini_et = EvoTensors(dim)
        new{typeof(ini_pd), typeof(ini_et)}(ini_pd, ini_sd, ini_et)
    end
end

#Used across all versions (in varying ways)
struct LagrangianWorkspaceVariables{N}
    fwd_position::Vector{Float64} 
    vector1::Vector{Float64} 
    vector2::Vector{Float64} 
    matrix1::Matrix{Float64}
    matrix2::Matrix{Float64} 
    r3_tensor1::Array{Float64, 3} 
    r3_tensor2::Array{Float64, 3} 
    adaptive_data::AdaptiveData{N}

    function LagrangianWorkspaceVariables(dim)
        fwd_position = zeros(Float64, dim) 
        vector1 = zeros(Float64, dim) 
        vector2 = zeros(Float64, dim) 
        matrix1 = zeros(Float64, dim, dim)
        matrix2 = zeros(Float64, dim, dim)
        r3_tensor1= zeros(Float64, dim, dim, dim)
        r3_tensor2= zeros(Float64, dim, dim, dim)
        #New state for adaptive data:
        st = SplitState(zeros(Float64, dim), zeros(Float64, dim), Base.RefValue{Int64}(1))
        ada = AdaptiveData{dim}(adaptive_state = st)
        new{dim}(fwd_position, vector1, vector2, matrix1, matrix2, r3_tensor1, r3_tensor2, ada)
    end
end
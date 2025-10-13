using LinearAlgebra, StaticArrays, DiffResults, ForwardDiff, OrdinaryDiffEq, Distributions, TensorOperations, Plots, Random, FastLapackInterface
#using BenchmarkTools, JET, Cthulhu

include("pdmp definitions.jl")
include("sampling algorithm.jl")
include("transition functions.jl")
include("Flow evaluation/Linear position methods/evaluation.jl")
include("PDMPs/BPS/bps-type.jl")
include("PDMPs/Lagrangian-type pdmps/lagrangian-type.jl")
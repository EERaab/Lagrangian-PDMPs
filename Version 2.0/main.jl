using LinearAlgebra, StaticArrays, DiffResults, ForwardDiff, OrdinaryDiffEq, Distributions, TensorOperations, Plots, Random, FastLapackInterface
#using BenchmarkTools

include("pdmp definitions.jl")
include("sampling algorithm.jl")
include("transition functions.jl")
include("Flow evaluation/Linear position methods/evaluation.jl")
include("PDMPs/BPS/bps-type.jl")
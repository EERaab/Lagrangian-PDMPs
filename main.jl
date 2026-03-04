using LinearAlgebra, StaticArrays, DiffResults, ForwardDiff, OrdinaryDiffEq, Distributions, Plots, Random, FastLapackInterface, Accessors, RecursiveArrayTools, OrdinaryDiffEqCore, DataStructures
#using BenchmarkTools, JET, Cthulhu

include("pdmp definitions.jl")
include("sampling algorithm.jl")
include("transition functions.jl")
include("Flow evaluation/Linear position methods/evaluation.jl")
include("PDMPs/BPS/bps-type.jl")
include("PDMPs/Lagrangian-type pdmps/lagrangian-type.jl")
include("Flow evaluation/velocity ODE.jl")
include("Visualization/plotting.jl")
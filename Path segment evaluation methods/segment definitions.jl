
#If we are done with evolution then it is sometimes useful to set the new type to be a special type.
struct Terminal<:DynType
end

#We can update positions...
abstract type PositionDyn<:DynType
end

#...and we can update velocities
abstract type VelocityDyn<:DynType
end


#A large swath of dynamics are position-velocity dynamics, for which we update the position x according to dx = vdt.
#For now this is the only position dynamic we consider.
#In principle we could consider HMC where the DynType could be PositionMomentum instead.
struct PositionVelocity<:PositionDyn
end

#Velocity updates come in many forms. Here we assume the velocities are updated according to either a gradient reflection or a velocity ODE.
#In principle there may be many forms of gradient reflection updates for a single PDMP method.
#If we encounter such a PDMP we shall generalize to have the parameter N (an integer) in the struct
struct VelocityODE{K<:PDMP_Method}<:VelocityDyn
end

#Gradient reflection generalize the BPS jump kernel.
struct GradientReflection{K<:PDMP_Method}<:VelocityDyn 
end


include("ODE-based velocity methods/velocity ODE.jl")
include("Velocity-based position methods/evaluation.jl")
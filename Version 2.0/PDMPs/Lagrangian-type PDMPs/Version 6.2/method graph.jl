
#Vertices correspond to non-instant evolution (this is a computational notion - mathematically we can view all nearly evolutions as instantaneous)
position_evo_v62 = PathSegmentValues{2}(dyn = PositionVelocity())
velocity_evo_v62 = PathSegmentValues{1}(dyn = VelocityODE())
vertices = [position_evo_v62, velocity_evo_v62]

#We store the edges in the following way
#edges[1] = edges that start in 1 = an array of edges, each with an associated transition
#edges[2] = edges that start in 2 = an array of edges, each with an associated transition
#The order of the edges in edges[i] is such that it corresponds to the order in the fwd rates of the PathSegmentValues of vertex i.
#That is to say edges[i][j] = is the event that we transition from i through the j:th transition.
#The resulting vertex is thus not obvious 
xx_edge = (1, GradientReflection()) #the xx edge is a gradient reflection from position evo to position evo
xv_edge = (2, Identity())
vx_edge = (1, Identity())
Lagrangian_Graph = MethodDiGraph(vertices, edges)


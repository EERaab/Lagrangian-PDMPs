@kwdef struct SplitVelocity<:Lagrangian_Method
    hardness::Float64 = 1.0
end


function generate_pdmp_graph(method::Version6_2, target::TargetData{F})::PDMP_DiGraph where F
    dim = target.dimension

    #Vertices
        #We have N = dim different vertices corresponding to N different velocity evolutions.
        #Note that in velocity evolution of the j:th component the j:th rate is the probability to go to
        #velocity evolution.
        vertices = [MethodVertex(i, dim) for i in 1:dim]
        dynamics = [SplitVelocityFlow(i) for i in 1:dim]
        #We have another evolution for the position
        #In this version we only take 1 evolution (no BPS-type events)
        push!(vertices, MethodVertex(n+1, 1))
        push!(dynamics, PositionVelocity())
    ####

    #Edges
        edges = []
        #Technically we could implement the PDMP graph as a  (bi-directionally) complete (loop-free) digraph.
        #This is awfully tedious to construct for big dimensions, but whatever. 

        #The velocity edges are encoded as follows: The i→j edge is the j:th element of the i-edge list for j≠i
        #The i:th element of the i-edge list is the transformation into position.
    
            for i in 1:dim
                i_to_j_edges = MethodEdge[]
                for j in 1:dim
                    if j ≠ i
                        push!(i_to_j_edges, MethodEdge(i, j, 1))
                    else
                        push!(i_to_j_edges, MethodEdge(i, dim + 1, 1))
                    end
                end
                push!(edges, i_to_j_edges)
            end
        ####

        #Position edges 
            pos_edges = MethodEdge[]
            for i in 1:dim
                push!(pos_edges, MethodEdge(dim + 1, i, 1))
            end
            push!(edges, pos_edges)
        ####
    ####
    return PDMP_DiGraph(vertices, edges, dynamics, (Identity(),))
end
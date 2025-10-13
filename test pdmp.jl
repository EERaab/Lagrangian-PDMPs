include("pdmp definitions.jl")
struct test_method <:PDMP_Method
    test_field
end

target(N) = TargetData(x->sin(sum(x.^2)), N)

function generate_pdmp_graph(method::test_method, target::TargetData)
    vertex_position = MethodVertex(PositionVelocity(), 1)
    vertices = Union{MethodVertex{VelocityPartial}, MethodVertex{PositionVelocity}}[vertex_position]

    bps_edge = MethodEdge(1, 1, GradientReflection())
    edges = Vector{Any}[[bps_edge]]
    for i in 1:target.dimension
        push!(vertices, MethodVertex(VelocityPartial(i), 2))

        #The edge from pos to vel_i
        pos_vel_i_edge = MethodEdge(1, i+1, Identity())
        push!(edges[1], pos_vel_i_edge)

        #The edge from vel_i to pos
        vel_i_edges = [MethodEdge(i+1, 1, Identity())]
        for j in 1:target.dimension
            if i ≠ j
                push!(vel_i_edges, MethodEdge(i+1,j+1, Identity()))
            end
        end
        push!(edges, vel_i_edges)
    end
    position_evolution_segment = Segment{1}()
    velocity_evolution_segment = Segment{target.dimension}()
    segments = [position_evolution_segment, velocity_evolution_segment]
    return PDMP_DiGraph(vertices, edges, segments)
end

n = 10
test_pdmp = PDMP(method = test_method(2), target = target(n))
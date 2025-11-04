function visualize_2D_density(log_pdfunc::Function; xs = range(-3,3,length=100), ys = range(-3,3,length=100), heat::Bool = false)
    q(x,y) = exp(log_pdfunc([x, y]))
    if !heat
        return surface(xs, ys, q, zlims = (0,1), legend = false)
    else
        return heatmap(xs,ys,q, legend = false)
    end
end

function plot_algorithm_run(pdmp::PDMP, x_marginal::Function, 
    y_marginal::Function, initial_state::SplitState, numerics::NumericalParameters; max_point_attempts::Int = 10, max_time::Float64 = 1.0,
    only_marginal::Bool = false, verbose::Bool = true)  
      
    samples, acceptances = algorithm(pdmp, numerics, initial_state = initial_state, max_point_attempts = max_point_attempts, max_time = max_time);
    st = stack(samples);
    L = length(unique(samples))
    if verbose
        println("Number of unique points: $L")
        acc = plot(acceptances, label = "Acceptance rate")
    end

    x_marginal_plot = histogram(st[1,:], normalize = :pdf, bins = max(50, min(100, div(L, 30))), legend=false)
    plot!(x_marginal)

    y_marginal_plot = histogram(st[2,:], normalize = :pdf, bins = max(50, min(100, div(L, 30))), legend=false)
    plot!(y_marginal)
    if !only_marginal
        hist_2D = histogram2d(st[1,:], st[2,:], bins = 100, legend = false)
        return plot(hist_2D, x_marginal_plot, y_marginal_plot, layout = @layout [a ; b  c])
    end
    title = string(numerics.position_method)*" for "*string(pdmp.method)*" on $L out of"*string(point_number)*" points"
    if verbose
        return plot(x_marginal_plot, y_marginal_plot, acc, plot_title = title, plot_titlefontsize = 10, layout = @layout [a b; c])
    end
    return plot(x_marginal_plot, y_marginal_plot, plot_title = title, plot_titlefontsize = 10, layout = @layout [b c])
end

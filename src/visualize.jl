## Scripts for model visualization ##


function plot_reconstruction(
        r::CNMF_results,
        t_range::UnitRange;
        sort_units=true,
    )

    idx = sort_units ? sortperm(r) : 1:num_units(r)

    # Form model estimate
    est = tensor_conv(r.W, r.H)

    fig, ax = plt.subplots(2, 1)
    ax[1].imshow(r.data[idx, t_range])
    ax[2].imshow(est[idx, t_range])

    return fig, ax
end


function plot_Ws(
    r::CNMF_results;
    sort_units=true,
    trueW=nothing
)

    data, W, H = r.data, r.W, r.H
    idx = sort_units ? sortperm(r) : 1:num_units(r)

    fig, ax = plt.subplots(1, num_components(r))
    for (k, a) in enumerate(ax)
        format_imshow_axis(a)
        a.imshow(transpose(W[:, idx, k]), aspect="auto")
    end

    if !(trueW === nothing)
        fig2, ax2 = plt.subplots(1, num_components(r))
        for (k, a) in enumerate(ax2)
            format_imshow_axis(a)
            a.imshow(transpose(trueW[:, idx, k]), aspect="auto")
        end
    end

    return fig, ax
end

function format_imshow_axis(
                            ax; 
                            border_width=1, 
                            border_color="black",
                            remove_ticks=true
                            )
    if remove_ticks
        ax.set_yticks([])
        ax.set_xticks([])
    end
    for side in ["left", "right", "bottom", "top"]
        ax.spines[side].set_linewidth(border_width)
        ax.spines[side].set_color(border_color)
    end
end

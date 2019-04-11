## Scripts for model visualization ##
import PyPlot; const plt = PyPlot


function plot_result(result::CNMF_results; tmin=0, tmax=-1)

    # Unpack results object.
    data, W, H = result.data, result.W, result.H
    est = tensor_conv(W, H)

    # Create figure and axes for plotting data.
    fig1, ax = plt.subplots(2, 1)
    ax[1].imshow(data)
    ax[2].imshow(est)

    fig2, ax = plt.subplots(1, num_components(result))
    for (k, a) in enumerate(ax)
        a.imshow(transpose(W[:, :, k]), aspect="auto")
    end

    return fig1
end
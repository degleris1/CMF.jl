using PyCall
include("model.jl")
py"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_result(data, W, H, tmin=0, tmax=-1, outer_pad=.05, inner_pad=.05,
                data_ax_height=.7, data_ax_width=.7, figsize=(10, 6)):
    
    # Truncate data and H to desired window.
    data = data[:, tmin:tmax]
    H = H[:, tmin:tmax]
    num_components = H.shape[0]

    # Layout parameters for figure.
    h_ax_height = 1 - data_ax_height
    w_ax_width = 1 - data_ax_width
    pad = inner_pad + outer_pad

    # Create figure and axes for plotting data.
    fig = plt.figure(figsize=figsize)
    data_ax_pos = {
        "left": w_ax_width + inner_pad * w_ax_width,
        "bottom": outer_pad,
        "right": 1.0 - outer_pad,
        "top": data_ax_height - inner_pad * h_ax_height,
    }
    data_ax = plt.subplot(GridSpec(1, 1, **data_ax_pos)[0])

    data_ax.set_xticks([])
    data_ax.set_yticks([])

    # Set up axes for visualizing model motifs.
    w_ax = []
    w_ax_pos = {
        "left": outer_pad,
        "bottom": outer_pad,
        "right": w_ax_width,
        "top": data_ax_height - inner_pad * h_ax_height,
        "wspace": inner_pad,
    }
    for gs in GridSpec(1, num_components, **w_ax_pos):
        w_ax.append(plt.subplot(gs))

    # for ax in w_ax[1:]:
    #     ax.get_shared_x_axes().join(w_ax[0], ax)
    #     ax.get_shared_y_axes().join(w_ax[0], ax)
    for ax in w_ax:
        ax.set_yticks([])
        ax.set_xticks([])

    # Set up axes for visualizing motif times.
    h_ax = []
    h_ax_pos = {
        "left": w_ax_width + inner_pad * w_ax_width,
        "bottom": data_ax_height,
        "right": 1 - outer_pad,
        "top": 1 - outer_pad,
        "hspace": inner_pad,
    }
    for gs in GridSpec(num_components, 1, **h_ax_pos):
        h_ax.append(plt.subplot(gs))

    # for ax in h_ax[1:]:
    #     ax.get_shared_x_axes().join(w_ax[0], ax)
    #     ax.get_shared_y_axes().join(w_ax[0], ax)
    for ax in h_ax:
        ax.set_yticks([])
        ax.set_xticks([])

    # Plot data
    data_ax.imshow(data, aspect='auto')

    # Plot timing factors.
    for ax, h in zip(h_ax, H):
        ax.plot(h)
        ax.set_xlim([0, len(h)])
        ax.axis('off')

    # Plot motifs.
    for ax, w in zip(w_ax, W.T):
        ax.imshow(w, aspect='auto')
    
    plt.show()
    return f
"""

function plot_result(result_struct::CNMF_results)
    return py"plot_result"(result_struct.data, result_struct.W, result_struct.H)
end
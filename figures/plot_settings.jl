# PLOT SETTINGS
doc_textwidth = 469.75
fontsize = 10
ticksize = 8

plt.matplotlib.pyplot.style.use("seaborn")
plt.rc("text", usetex=true)
plt.rc("font", family="serif", size=fontsize)
plt.rc("lines", linewidth=1.5)
plt.rc("figure", titlesize=fontsize)
plt.rc("axes", labelsize=fontsize, titlesize=fontsize)
plt.rc("legend", fontsize=ticksize)
plt.rc("xtick", labelsize=ticksize)
plt.rc("ytick", labelsize=ticksize) 
plt.rc("lines", linestyle="-.")


"""
    Set aesthetic figure dimensions to avoid scaling in latex.
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    aspect: float or string
            If provided as a string, must be equal to the default, "golden".
            This sets the aspect ratio to be the golden ratio.
            Otherwise, this should be a float specifying the aspect ratio.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
"""
function set_size(width; fraction=1, aspect="golden")
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5^(.5) - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    if aspect == "golden"
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio
    else
        fig_height_in = fig_width_in * aspect
    end

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
end

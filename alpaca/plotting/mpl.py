import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 12, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})
import numpy as np
from ..statistics.functions import nsigmas
from .palettes import darker_set3, trafficlights
from ..statistics.chisquared import ChiSquared, combine_chi2

def exclusionplot(x: np.ndarray[float], y: np.ndarray[float], chi2: list[ChiSquared] | ChiSquared, xlabel: str, ylabel: str, title: str | None = None, ax: plt.Axes | None = None, global_chi2: ChiSquared | None = None) -> plt.Axes:
    """
    Create an exclusion plot.

    Parameters
    ----------
    x : np.ndarray[float]
        The x-coordinates of the data points.
    y : np.ndarray[float]
        The y-coordinates of the data points.
    chi2 : list[ChiSquared]
        A list of ChiSquared objects representing the exclusion regions.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str | None, optional
        The title of the plot (default is None).
    ax : plt.Axes | None, optional
        The matplotlib Axes object to plot on (default is None, which creates a new figure).
    global_chi2 : ChiSquared | None, optional
        A ChiSquared object representing the global exclusion significance (default is None, which uses the combined chi squared).

    """
    cmap_trafficlights = ListedColormap(trafficlights+['#000000'])
    colors = darker_set3*4
    lss = ['solid']*len(darker_set3) + ['dashed']*len(darker_set3) + ['dotted']*len(darker_set3) + ['dashdot']*len(darker_set3)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    else:
        fig = ax.get_figure()
    legend_elements = ax.get_legend_handles_labels()[0]
    if isinstance(chi2, ChiSquared) and global_chi2 is None:
        global_chi2 = chi2
        chi2 = []
    elif isinstance(chi2, ChiSquared) and global_chi2 is not None:
        chi2 = [chi2]
    elif global_chi2 is None:
        global_chi2 = combine_chi2(chi2, 'Global', 'Global', 'Global')
    pl = ax.contourf(x,y, global_chi2.significance(), levels=list(np.linspace(0, 5, 150)), cmap=cmap_trafficlights, vmax=5, extend='max', zorder=-20)
    
    i_color = 0
    i_ls = 0
    for c in chi2:
        sigmas = c.significance()
        if np.all(np.isnan(sigmas)):
            continue
        if np.nanmax(sigmas) < 2:
            continue
        mask = np.nan_to_num(sigmas)
        if c.sector.color is not None:
            color = c.sector.color
        else:
            color = colors[i_color]
            i_color += 1
        if c.sector.ls is not None:
            ls = c.sector.ls
        else:
            ls = lss[i_ls]
            i_ls += 1
        if c.sector.lw is not None:
            lw = c.sector.lw
        else:
            lw = 1.0
        ax.contour(x, y, mask, levels=[2], colors = color, linestyles=ls, linewidths=lw)
        legend_elements.append(plt.Line2D([0], [0], color=color, ls=ls, label=c.sector.tex, lw=lw))
    ax.set_xscale('log')
    ax.set_yscale('log')
    cb = fig.colorbar(pl, extend='max')
    cb.set_label(r'Exclusion significance [$\sigma$]')
    cb.set_ticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=9, fontsize=8)
    plt.tight_layout()

    ax.set_rasterization_zorder(-10)
    return ax
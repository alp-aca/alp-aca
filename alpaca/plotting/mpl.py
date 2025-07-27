import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
import distinctipy
from collections.abc import Container
plt.rcParams.update({'font.size': 12, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})
import numpy as np
from .palettes import trafficlights
from ..statistics.chisquared import ChiSquared, combine_chi2
from ..decays.decays import to_tex
from ..biblio import citations

ref_matplotlib = r'''@Article{Hunter:2007,
  Author    = {Hunter, J. D.},
  Title     = {Matplotlib: A 2D graphics environment},
  Journal   = {Computing in Science \& Engineering},
  Volume    = {9},
  Number    = {3},
  Pages     = {90--95},
  abstract  = {Matplotlib is a 2D graphics package used for Python for
  application development, interactive scripting, and publication-quality
  image generation across user interfaces and operating systems.},
  publisher = {IEEE COMPUTER SOC},
  doi       = {10.1109/MCSE.2007.55},
  year      = 2007
}'''

def exclusionplot(x: Container[float], y: Container[float], chi2: list[ChiSquared] | ChiSquared, xlabel: str, ylabel: str, title: str | None = None, ax: Axes | None = None, global_chi2: ChiSquared | bool = True) -> Axes:
    """
    Create an exclusion plot.

    Parameters
    ----------
    x : Container[float]
        The x-coordinates of the data points.
    y : Container[float]
        The y-coordinates of the data points.
    chi2 : list[ChiSquared] | ChiSquared
        The ChiSquared object(s) representing the exclusion regions. If a single ChiSquared object is provided, it will be treated as the global exclusion significance.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str | None, optional
        The title of the plot (default is None).
    ax : plt.Axes | None, optional
        The matplotlib Axes object to plot on (default is None, which creates a new figure).
    global_chi2 : ChiSquared | bool, optional
        A ChiSquared object representing the global exclusion significance (default is True, which uses the combined chi squared).
        If set to False, no global significance will be plotted.

    """
    citations.register_bibtex('matplotlib', ref_matplotlib)
    cmap_trafficlights = ListedColormap(trafficlights+['#000000'])
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    else:
        fig = ax.get_figure()
    legend_elements = ax.get_legend_handles_labels()[0]
    if isinstance(chi2, ChiSquared):
        if global_chi2 is True:
            global_chi2 = chi2
            chi2 = []
        else:
            chi2 = [chi2]
    elif global_chi2 is True:
        global_chi2 = combine_chi2(chi2, 'Global', 'Global', 'Global')
    if isinstance(global_chi2, ChiSquared):
        pl = ax.contourf(x,y, global_chi2.significance(), levels=list(np.linspace(0, 5, 150)), cmap=cmap_trafficlights, vmax=5, extend='max', zorder=-20)
    
    colors_used = ['#ffffff', '#bfff86', '#fffd66', '#f06060', '#68292a', '#000000']
    chi2_contours = []
    for c in chi2:
        sigmas = c.significance()
        if np.all(np.isnan(sigmas)):
            continue
        if np.nanmax(sigmas) < 2:
            continue
        chi2_contours.append(c)
        if c.sector.color is not None:
            colors_used.append(c.sector.color)
    hex2rgb = lambda hex: tuple(int(hex[i:i+2], 16)/255 for i in (1, 3, 5))
    palette = distinctipy.get_colors(len(chi2_contours)- (len(colors_used)-6), exclude_colors=[hex2rgb(c) for c in colors_used], pastel_factor=0.7)
    i_color = 0
    for c in chi2_contours:
        sigmas = c.significance()
        mask = np.nan_to_num(sigmas)
        if c.sector.color is not None:
            color = c.sector.color
        else:
            color = palette[i_color]
            i_color += 1
        if c.sector.ls is not None:
            ls = c.sector.ls
        else:
            ls = 'solid'
        if c.sector.lw is not None:
            lw = c.sector.lw
        else:
            lw = 1.0
        ax.contour(x, y, mask, levels=[2], colors = color, linestyles=ls, linewidths=lw)
        legend_elements.append(plt.Line2D([0], [0], color=color, ls=ls, label=c.sector.tex, lw=lw))
    ax.set_xscale('log')
    ax.set_yscale('log')
    if isinstance(global_chi2, ChiSquared):
        cb = fig.colorbar(pl, extend='max')
        cb.set_label(r'Exclusion significance [$\sigma$]')
        cb.set_ticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=12)
    if len(legend_elements) > 0:
        ax.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=9, fontsize=8)
    plt.tight_layout()

    ax.set_rasterization_zorder(-10)
    return ax

def alp_channels_plot(x: Container[float], channels: dict[str, Container[float]], xlabel: str, ylabel: str, ymin: float | None = None, title: str | None = None, ax: Axes | None = None) -> Axes:
    """
    Create a plot for ALP decay channels.

    Parameters
    ----------
    x : Container[float]
        The x-coordinates of the data points.
    channels : dict[str, Container[float]]
        A dictionary where keys are channel names and values are the corresponding y-coordinates.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    ymin : float
        The minimum value for the y-axis.
    ymax : float
        The maximum value for the y-axis.
    title : str | None, optional
        The title of the plot (default is None).
    ax : plt.Axes | None, optional
        The matplotlib Axes object to plot on (default is None, which creates a new figure).

    """
    citations.register_bibtex('matplotlib', ref_matplotlib)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()
    
    if ymin is None:
        ncols = len(channels)
    else:
        ncols = sum(1 for channel in channels if np.max(channels[channel]) > ymin)
    palette = distinctipy.get_colors(ncols, pastel_factor=0.7)
    for channel, y in channels.items():
        if (ymin is None) or (np.max(y) > ymin):
            ax.loglog(x, y, label=to_tex(channel), color=palette.pop(0))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if title is not None:
        ax.set_title(title)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=2, fontsize=8)
    plt.tight_layout()
    
    return ax
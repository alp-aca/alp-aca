'''
alpaca.plotting.mpl
======================

This module contains functions to handle plotting with Matplotlib.

Functions
---------

exclusionplot
    Create an exclusion plot.

alp_channels_plot
    Create a plot for ALP decay channels.
'''

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
import matplotlib.image as mpimg
import distinctipy
from collections.abc import Container
plt.rcParams.update({'font.size': 12, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})
import numpy as np
from .palettes import trafficlights
from ..statistics.chisquared import ChiSquared, combine_chi2
from ..decays.decays import to_tex
from ..biblio import citations
from ..scan import Axis
import os

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

def add_logo_avoiding_legend(fig, ax, logo_path, logo_size=0.15, margin=0.02, alpha=0.8, position=0):
    """
    Add a logo inside the axes, automatically avoiding the legend.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    logo_path : str
        Path to the logo image file
    logo_size : float
        Size of the logo as fraction of axes size (default: 0.15)
    margin : float
        Margin from axes edges (default: 0.02)
    """
    
    # Load the logo
    logo = mpimg.imread(logo_path)
    
    # Calculate logo dimensions maintaining aspect ratio
    logo_height, logo_width = logo.shape[:2]
    aspect_ratio = logo_width / logo_height
    logo_width_axes = logo_size * aspect_ratio
    logo_height_axes = logo_size
    
    # Get legend position if it exists
    legend = ax.get_legend()
    
    # Define potential positions (x, y in axes coordinates)
    candidate_positions: dict[str, tuple[float, float]] = {
        'upper right': (1 - margin - logo_width_axes, 1 - margin - logo_height_axes),
        'upper left': (margin, 1 - margin - logo_height_axes),
        'lower right': (1 - margin - logo_width_axes, margin),
        'lower left': (margin, margin),
    }

    positions_num = {
        0: 'best',
        1: 'upper right',
        2: 'upper left',
        3: 'lower left',
        4: 'lower right'
    }
    
    if isinstance(position, int) and position in positions_num.keys():
        position = positions_num[position]
    elif not position in positions_num.values():
        raise ValueError(f"Unkown option for logo placement {position}")

    if position in candidate_positions.keys():
        logo_x, logo_y = candidate_positions[position]
    elif legend is not None:
        # Force drawing to get accurate legend position
        fig.canvas.draw()
        
        # Get legend bounding box in axes coordinates
        legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
        legend_bbox_axes = legend_bbox.transformed(ax.transAxes.inverted())
        
        # Find the position with maximum distance from legend
        best_position = None
        max_distance = -1
        
        for logo_x, logo_y in candidate_positions.values():
            # Calculate logo center
            logo_center_x = logo_x + logo_width_axes / 2
            logo_center_y = logo_y + logo_height_axes / 2
            
            # Calculate legend center
            legend_center_x = (legend_bbox_axes.x0 + legend_bbox_axes.x1) / 2
            legend_center_y = (legend_bbox_axes.y0 + legend_bbox_axes.y1) / 2
            
            # Calculate distance between centers
            distance = np.sqrt((logo_center_x - legend_center_x)**2 + 
                             (logo_center_y - legend_center_y)**2)
            
            # Check if logo would overlap with legend
            logo_bbox_x0, logo_bbox_x1 = logo_x, logo_x + logo_width_axes
            logo_bbox_y0, logo_bbox_y1 = logo_y, logo_y + logo_height_axes
            
            overlaps = not (logo_bbox_x1 < legend_bbox_axes.x0 or 
                          logo_bbox_x0 > legend_bbox_axes.x1 or
                          logo_bbox_y1 < legend_bbox_axes.y0 or 
                          logo_bbox_y0 > legend_bbox_axes.y1)
            
            # Prefer positions that don't overlap and have maximum distance
            if not overlaps and distance > max_distance:
                max_distance = distance
                best_position = (logo_x, logo_y)
            elif best_position is None:  # All positions overlap, choose farthest
                if distance > max_distance:
                    max_distance = distance
                    best_position = (logo_x, logo_y)
        
        logo_x, logo_y = best_position
    else:
        # No legend, default to upper right
        logo_x = 1 - margin - logo_width_axes
        logo_y = 1 - margin - logo_height_axes
    
    # Add logo using inset axes
    logo_ax = ax.inset_axes([logo_x, logo_y, logo_width_axes, logo_height_axes],
                            transform=ax.transAxes, zorder=-10)
    
    # Disable clipping to prevent cropping
    logo_ax.set_clip_on(False)
    
    # Display the image
    im = logo_ax.imshow(logo, alpha=alpha)
    im.set_clip_on(False)
    
    logo_ax.axis('off')
    
    return logo_ax

def exclusionplot(x: Container[float] | Axis, y: Container[float] | Axis, chi2: list[ChiSquared] | ChiSquared, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None, ax: Axes | None = None, global_chi2: ChiSquared | bool = True, logo_position = 0) -> Axes:
    """
    Create a static exclusion plot.

    Parameters
    ----------
    x : Container[float] | Axis
        The x-coordinates of the data points.
    y : Container[float] | Axis
        The y-coordinates of the data points.
    chi2 : list[ChiSquared] | ChiSquared
        The ChiSquared object(s) representing the exclusion regions. If a single ChiSquared object is provided, it will be treated as the global exclusion significance.
    xlabel : str | None, optional
        The label for the x-axis.
    ylabel : str | None, optional
        The label for the y-axis.
    title : str | None, optional
        The title of the plot (default is None).
    ax : plt.Axes | None, optional
        The matplotlib Axes object to plot on (default is None, which creates a new figure).
    global_chi2 : ChiSquared | bool, optional
        A ChiSquared object representing the global exclusion significance (default is True, which uses the combined chi squared).
        If set to False, no global significance will be plotted.
    logo_position : int | str | None, optional
        The location of the ALP-aca logo on the plot (default is 0, which corresponds to the upper right corner). Valid values are:
            * 0 or 'best': avoids legend
            * 1 or 'upper right': upper right corner
            * 2 or 'upper left': upper left corner
            * 3 or 'lower left': lower left corner
            * 4 or 'lower right': lower right corner
            * None: no logo will be added

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
    if isinstance(x, Axis):
        x0 = x.values
    else:
        x0 = x
    if isinstance(y, Axis):
        y0 = y.values
    else:
        y0 = y
    if isinstance(global_chi2, ChiSquared):
        pl = ax.contourf(x0,y0, global_chi2.significance(), levels=list(np.linspace(0, 5, 150)), cmap=cmap_trafficlights, vmax=5, extend='max', zorder=-20)
    
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
        ax.contour(x0, y0, mask, levels=[2], colors = color, linestyles=ls, linewidths=lw)
        legend_elements.append(plt.Line2D([0], [0], color=color, ls=ls, label=c.sector.tex, lw=lw))
    ax.set_xscale('log')
    ax.set_yscale('log')
    if isinstance(global_chi2, ChiSquared):
        cb = fig.colorbar(pl, extend='max')
        cb.set_label(r'Exclusion significance [$\sigma$]')
        cb.set_ticks([0, 1, 2, 3, 4, 5])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif isinstance(x, Axis):
        ax.set_xlabel(x.tex)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif isinstance(y, Axis):
        ax.set_ylabel(y.tex)
    if title is not None:
        ax.set_title(title, fontsize=12)
    if len(legend_elements) > 0:
        ax.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=9, fontsize=8)
    plt.tight_layout()

    if logo_position is not None:
        current_dir = os.path.dirname(__file__)
        logo_path = os.path.join(current_dir, 'logo.png')
        ax = add_logo_avoiding_legend(fig, ax, logo_path, 0.07, position=logo_position)

    ax.set_rasterization_zorder(-10)
    return ax

def alp_channels_plot(x: Container[float] | Axis, channels: dict[str, Container[float]], xlabel: str | None = None, ylabel: str | None = None, ymin: float | None = None, title: str | None = None, ax: Axes | None = None, logo_position = 0) -> Axes:
    """
    Create a static plot for ALP decay channels.

    Parameters
    ----------
    x : Container[float] | Axis
        The x-coordinates of the data points.
    channels : dict[str, Container[float]]
        A dictionary where keys are channel names and values are the corresponding y-coordinates.
    xlabel : str | None, optional
        The label for the x-axis.
    ylabel : str | None, optional
        The label for the y-axis.
    ymin : float
        The minimum value for the y-axis.
    title : str | None, optional
        The title of the plot (default is None).
    ax : plt.Axes | None, optional
        The matplotlib Axes object to plot on (default is None, which creates a new figure).
    logo_position : int | str | None, optional
        The location of the ALP-aca logo on the plot (default is 0, which corresponds to the upper right corner). Valid values are:
            - 0 or 'best': avoids legend
            - 1 or 'upper right': upper right corner
            - 2 or 'upper left': upper left corner
            - 3 or 'lower left': lower left corner
            - 4 or 'lower right': lower right corner
            - None: no logo will be added

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
    if isinstance(x, Axis):
        x_vals = x.values
    else:
        x_vals = x
    for channel, y in channels.items():
        if (ymin is None) or (np.max(y) > ymin):
            ax.loglog(x_vals, y, label=to_tex(channel), color=palette.pop(0))

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif isinstance(x, Axis):
        ax.set_xlabel(x.tex)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if title is not None:
        ax.set_title(title)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=2, fontsize=8)
    plt.tight_layout()


    if logo_position is not None:
        current_dir = os.path.dirname(__file__)
        logo_path = os.path.join(current_dir, 'logo.png')
        ax = add_logo_avoiding_legend(fig, ax, logo_path, 0.07, position=logo_position)
    
    return ax
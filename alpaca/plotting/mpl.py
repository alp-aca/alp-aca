import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 12, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})
import numpy as np
from ..statistics.functions import nsigmas
from .palettes import darker_set3, trafficlights
from ..statistics.chisquared import ChiSquared, combine_chi2

def exclusionplot(x: np.ndarray[float], y: np.ndarray[float], chi2: list[ChiSquared], xlabel: str, ylabel: str, title: str, ax=None):
    cmap_trafficlights = ListedColormap(trafficlights+['#000000'])
    colors = darker_set3*4
    lss = ['solid']*len(darker_set3) + ['dashed']*len(darker_set3) + ['dotted']*len(darker_set3) + ['dashdot']*len(darker_set3)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    legend_elements = []
    global_chi2 = combine_chi2(chi2, 'Global', 'Global', 'Global')
    pl = plt.contourf(x,y, global_chi2.significance(), levels=list(np.linspace(0, 5, 150)), cmap=cmap_trafficlights, vmax=5, extend='max')
    
    i = 0
    for c in chi2:
        sigmas = c.significance()
        if np.nanmax(sigmas) < 2:
            continue
        mask = np.nan_to_num(sigmas)
        plt.contour(x, y, mask, levels=[2], colors = colors[i], linestyles=lss[i])
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], ls=lss[i], label=c.sector.tex))
        i += 1
    ax.set_xscale('log')
    ax.set_yscale('log')
    cb = plt.colorbar(pl, extend='max')
    cb.set_label(r'Exclusion significance [$\sigma$]')
    cb.set_ticks([0, 1, 2, 3, 4, 5])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12)
    plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=9, fontsize=8)
    plt.tight_layout()

    return ax
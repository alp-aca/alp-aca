import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 12, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})
import numpy as np
from ..statistics.functions import nsigmas
from .palettes import darker_set3, newmagma

def exclusionplot(x, y, chi2, xlabel, ylabel, title, tex):
    def max_finite(x):
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return -1
        return np.max(finite)
    cmap_newmaga = ListedColormap(newmagma+['#000000'])
    colors = {k: c for k, c in zip(chi2.keys(), darker_set3)}
    fig, ax = plt.subplots()
    legend_elements = []
    pl = plt.contourf(x,y, nsigmas(chi2[('', 'Global')],2), levels=list(np.linspace(0, 3, 100)), cmap=cmap_newmaga, vmax=3, extend='max')
    for observable, chi2_obs in chi2.items():
        if observable == ('', 'Global'):
            continue
        if max_finite(nsigmas(chi2_obs, 2)) < 2:
            continue
        mask = np.where(np.isnan(chi2_obs), 0, nsigmas(chi2_obs, 2))
        plt.contour(x, y, mask, levels=[2], colors = colors[observable])
        legend_elements.append(plt.Line2D([0], [0], color=colors[observable], label=tex[observable[0]] + ' (' + observable[1] + ')'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    cb = plt.colorbar(pl, extend='max')
    cb.set_label(r'Exclusion significance [$\sigma$]')
    cb.set_ticks([0, 1, 2, 3])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=12)
    plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=9, fontsize=8)
    plt.tight_layout()

    return fig, ax
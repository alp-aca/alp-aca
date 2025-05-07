import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 12, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Computer Modern Roman'})
import numpy as np
from ..statistics.functions import nsigmas
from .palettes import darker_set3, trafficlights

def exclusionplot(x, y, chi2, xlabel, ylabel, title, tex, ax=None):
    def max_finite(x):
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return -1
        return np.max(finite)
    cmap_trafficlights = ListedColormap(trafficlights+['#000000'])
    colors = {k: c for k, c in zip(chi2.keys(), darker_set3*4)}
    styles_list = ['solid']*len(darker_set3) + ['dashed']*len(darker_set3) + ['dotted']*len(darker_set3) + ['dashdot']*len(darker_set3)
    lss = {k: s for k, s in zip(chi2.keys(), styles_list)}
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    legend_elements = []
    pl = plt.contourf(x,y, nsigmas(chi2[('', 'Global')][0],chi2[('', 'Global')][1]), levels=list(np.linspace(0, 5, 150)), cmap=cmap_trafficlights, vmax=5, extend='max')
    
    for observable, chi2_obs in chi2.items():
        if observable == ('', 'Global'):
            continue
        if max_finite(nsigmas(chi2_obs[0], chi2_obs[1])) < 2:
            continue
        mask = np.nan_to_num(nsigmas(chi2_obs[0], chi2_obs[1]))
        plt.contour(x, y, mask, levels=[2], colors = colors[observable], linestyles=lss[observable])
        if tex != None:
            if isinstance(observable, tuple):
                label = tex[observable[0]] + ' (' + observable[1] + ')'
            else:
                label = tex[observable]
            legend_elements.append(plt.Line2D([0], [0], color=colors[observable], ls=lss[observable], label=label))
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
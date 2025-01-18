import contourpy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from contourpy.util.bokeh_util import filled_to_bokeh
from bokeh.models import HoverTool, LinearColorMapper
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from itertools import cycle
from .palettes import newmagma, darker_set3
from ..statistics.functions import nsigmas

def exclusionplot(x, y, chi2, xlabel, ylabel, title, tex = None):
    def max_finite(x):
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return -1
        return np.max(finite)
    fig1 = figure(
        title=title,
        x_axis_label=xlabel,
        y_axis_label=ylabel,
        x_axis_type='log',
        y_axis_type='log',
        x_range=(np.min(x), np.max(x)),
        y_range=(np.min(y), np.max(y)),
        tools='box_zoom,reset,save,crosshair',
        height=600,
        width=1200
    )
    colors = {k: v for k, v in zip(chi2.keys(), cycle(darker_set3))}
    interp_global = RegularGridInterpolator((x, y), chi2[('', 'Global')].T, bounds_error=False, fill_value=np.nan)
    x_plot = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 2000)
    y_plot = np.logspace(np.log10(np.min(y)), np.log10(np.max(y)), 2000)
    X, Y = np.meshgrid(x_plot, y_plot)
    xx, yy = np.meshgrid(x, y)
    Z = interp_global((X, Y))
    cmap = LinearColorMapper(newmagma, low=0, high=3, nan_color='white', high_color='black')
    hvr = HoverTool(tooltips=[('observable', '@observable'), ('experiment', '@experiment')])
    fig1.add_tools(hvr)
    exclusion = nsigmas(Z,2)
    source = ColumnDataSource(data={'xs': [x_plot], 'ys':[y_plot], 'exclusion': [exclusion]})
    contour_global = fig1.image(image='exclusion', x=np.min(x), y=np.min(y), dh=np.max(y)-np.min(y), dw=np.max(x)-np.min(x), color_mapper=cmap, origin='bottom_left', source=source)
    colorbar  = contour_global.construct_color_bar(title = r'Exclusion significance [$$\sigma$$]', title_text_align='center', title_text_baseline='middle')
    fig1.add_layout(colorbar, 'right')
    contours_obs = []
    for observable, chi2_obs in chi2.items():
        if observable == ('', 'Global'):
            break
        if max_finite(nsigmas(chi2_obs, 2)) < 2:
            continue
        contour = contourpy.contour_generator(xx, yy, nsigmas(chi2_obs,2))
        contour_bokeh = filled_to_bokeh(contour.filled(2, np.inf), contour.fill_type)
        source = ColumnDataSource(data={'xs': [contour_bokeh[0]], 'ys':[contour_bokeh[1]], 'observable': [observable[0]], 'experiment': [observable[1]]})
        if isinstance(observable, tuple):
            label = tex[observable[0]] + ' (' + observable[1] + ')'
        else:
            label = tex[observable]
        contour_fig = fig1.multi_polygons(xs = 'xs', ys='ys', line_width=4, source=source, color=colors[observable], legend_label=label, fill_alpha=0, hover_alpha=0.3, visible=False)
        contours_obs.append(contour_fig)
    fig1.legend.click_policy = 'hide'
    fig1.add_layout(fig1.legend[0], 'right')
    hvr.renderers = contours_obs
    return fig1
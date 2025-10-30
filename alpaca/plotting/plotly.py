import plotly.graph_objects as go
import plotly

from .palettes import trafficlights
from ..statistics import ChiSquared, combine_chi2
from typing import Container
import distinctipy
import numpy as np
import jinja2
import os
from .. import __version__
from ..scan import Axis
from ..biblio import citations

ref_plotly = '''@software{Kruchten_An_interactive_open-source_2025,
author = {Kruchten, Nicolas and Seier, Andrew and Parmer, Chris},
doi = {10.5281/zenodo.14503524},
month = oct,
title = {{An interactive, open-source, and browser-based graphing library for Python}},
url = {https://github.com/plotly/plotly.py},
version = {6.3.1},
year = {2025}
}'''

def prepare_nb():
    from IPython.display import display, HTML

    plotly.offline.init_notebook_mode()
    display(HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))

def exclusionplot(x: Container[float] | Axis, y: Container[float] | Axis, chi2: list[ChiSquared] | ChiSquared, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None, fig: go.Figure | None = None, global_chi2: ChiSquared | bool = True, xvar: str = 'x', yvar: str = 'y', xunits: str = '', yunits: str = '') -> go.Figure:
    citations.register_bibtex('plotly', ref_plotly)
    if isinstance(chi2, ChiSquared):
        if global_chi2 is True:
            global_chi2 = chi2
            chi2 = []
        else:
            chi2 = [chi2]
    elif global_chi2 is True:
        global_chi2 = combine_chi2(chi2, 'Global', 'Global', 'Global')

    if fig is None:
        fig = go.Figure()

    if isinstance(x, Axis):
        x_values = x.values
    else:
        x_values = np.array(x)
        if len(x_values.shape) == 2:
            x_values = x_values[0, :]
    if isinstance(y, Axis):
        y_values = y.values
    else:
        y_values = np.array(y)
        if len(y_values.shape) == 2:
            y_values = y_values[:, 0]

    if isinstance(global_chi2, ChiSquared):
        fig.add_trace(
            go.Heatmap(
                x = x_values,
                y = y_values,
                z = global_chi2.significance(),
                connectgaps=True,
                name = global_chi2.sector.tex,
                hovertemplate = f'{xvar} = %{{x}} {xunits}<br>{yvar} = %{{y}} {yunits}<br>Significance: %{{z:.2}} &#963;<extra></extra>',
                zmin = 0,
                zmax = 5,
                colorbar = {'title': {'text': 'Significance [&#963;]', 'side': 'bottom'}, 'xpad': 0, 'orientation': 'h', 'yanchor': 'top', 'y': -0.1},
                zsmooth='best',
                colorscale= [[i/len(trafficlights), trafficlights[i]] for i in range(len(trafficlights)) if i%2==0] + [[1.0, trafficlights[-1]]],
                showlegend= True
            )
        )

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
        x_c, y_c = c.contour(x_values, y_values)
        if c.sector.color is not None:
            color = c.sector.color
        else:
            color = palette[i_color]
            i_color += 1
        if c.sector.ls in ['dashed', 'dash', '--']:
            ls = 'dash'
        elif c.sector.ls in ['dotted', 'dot', ':']:
            ls = 'dot'
        else:
            ls = 'solid'
        if c.sector.lw is not None:
            lw = c.sector.lw
        else:
            lw = 2.0
        if isinstance(color, (tuple, list)) and len(color) == 3:
            col_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
        elif isinstance(color, str):
            col_hex = color
        fig.add_trace(
            go.Scatter(
                x = x_c,
                y = y_c,
                mode = 'lines',
                name = c.sector.tex,
                showlegend=True,
                meta = {'name': c.sector.name},
                hovertemplate = f'<b>%{{meta.name}}</b><br>{xvar} = %{{x}} {xunits}<br>{yvar} = %{{y}} {yunits}<extra></extra>',
                line = {'shape' : 'spline', 'color': col_hex, 'dash': ls, 'width': lw},
            )
        )

    fig.update_xaxes(type='log', range=[np.log10(np.min(x_values)), np.log10(np.max(x_values))], exponentformat='power')
    fig.update_yaxes(type='log', range=[np.log10(np.min(y_values)), np.log10(np.max(y_values))], exponentformat='power')
    if xlabel is not None:
        fig.update_xaxes(title_text=xlabel)
    elif isinstance(x, Axis):
        fig.update_xaxes(title_text=x.tex)
    if ylabel is not None:
        fig.update_yaxes(title_text=ylabel)
    elif isinstance(y, Axis):
        fig.update_yaxes(title_text=y.tex)

    fig.update_layout(
        legend = dict(
            orientation = 'v',
            yanchor = 'top',
            xanchor = 'right',
            x = 1.2,
        ),
        title = {'text': title, 'automargin': False, 'pad': {'t': -50, 'b': -50}, 'yanchor': 'top'} if title is not None else None,
        margin = {'autoexpand': True, 'pad': 0.8, 'b': 150, 't': 60, 'r': 200, 'l': 10},
        autosize = True,
    )

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    return fig

def save_html(fig: go.Figure, filename: str, title: str, template: str = 'basic'):
    template_file = os.path.join(os.path.dirname(__file__), 'html_templates', f'{template}.html')
    with open(filename, 'w') as f_out:
        with open(template_file, 'r') as f_template:
            template_content = f_template.read()
        jinja_template = jinja2.Template(template_content)
        fig_html = plotly.io.to_html(fig, full_html=False, include_plotlyjs='cdn', include_mathjax='cdn', default_width="80%", div_id ="plotly-figure", default_height="800px")
        rendered_html = jinja_template.render(
            title=title,
            fig=fig_html,
            version=__version__
        )
        f_out.write(rendered_html)

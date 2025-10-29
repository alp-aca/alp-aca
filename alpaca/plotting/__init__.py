from .engines import check_plotting_engines

__all__ = ['check_plotting_engines', 'palettes']

available_engines = [engine for engine, av in check_plotting_engines().items() if av]
if 'matplotlib' in available_engines:
    from . import mpl
    __all__.append('mpl')
if 'plotly' in available_engines:
    from . import plotly
    __all__.append('plotly')

def __getattr__(name):
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module 'alpaca.plotting' has no attribute '{name}'")

def __dir__():
    return __all__
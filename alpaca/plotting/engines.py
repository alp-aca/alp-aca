from importlib import metadata

def check_plotting_engines():
    engines = ['matplotlib', 'plotly']
    available = {}
    for engine in engines:
        try:
            metadata.version(engine)
            available[engine] = True
        except metadata.PackageNotFoundError:
            available[engine] = False
    return available
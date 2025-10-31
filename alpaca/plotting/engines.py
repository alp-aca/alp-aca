from importlib import metadata

def check_plotting_engines():
    '''
    Check the availability of plotting engines.

    Returns
    -------
    available (dict)
        A dictionary with the availability status of each plotting engine.
    '''
    engines = ['matplotlib', 'plotly']
    available = {}
    for engine in engines:
        try:
            metadata.version(engine)
            available[engine] = True
        except metadata.PackageNotFoundError:
            available[engine] = False
    return available
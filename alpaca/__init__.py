"""ALPaca: The ALP Automatic Computing Algorithm

Modules
-------
models
    Contains the classes to define the UV models.
experimental_data
    Contains classes and functions to handle experimental data.
statistics
    Contains functions to handle statistics.
plotting
    Contains functions to handle plotting.

Classes
-------
ALPcouplings
    A class to represent the couplings of ALPs to SM particles.
ALPcouplingsEncoder
    A class to encode ALPcouplings objects into JSON format.
ALPcouplingsDecoder
    A class to decode JSON formatted ALPcouplings objects.

Functions
---------
decay_width
    Calculates the decay width of a particle.
branching_ratio
    Calculates the branching ratio of a particle.
cross_section
    Calculates the cross section of a process.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
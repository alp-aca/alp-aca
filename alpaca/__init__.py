"""ALPaca: The ALP Automatic Computing Algorithm

Modules
-------
uvmodels
    Contains the classes to define the UV models.
experimental_data
    Contains classes and functions to handle experimental data.
sectors
    Contains classes and functions to handle sectors of observables.
statistics
    Contains functions to handle statistics.
plotting
    Contains functions to handle plotting.
citations
    Contains functions to handle bibliographical references.

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
meson_mixing
    Calculates the value of a meson mixing observable.
alp_channels_decay_widths
    Calculates the decay widths for all ALP channels.
alp_channels_branching_ratios
    Calculates the branching ratios for all ALP channels.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
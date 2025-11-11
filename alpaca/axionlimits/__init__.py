"""alpaca.axionlimits

This module interfaces with the AxionLimits repository by Ciaran O'Hare.
https://github.com/cajohare/AxionLimits

Objects
--------

limits_electrons :
    AxionLimits object for electron coupling limits.
limits_electrons_proj :
    AxionLimits object for projected electron coupling limits.
limits_photons :
    AxionLimits object for photon coupling limits.
limits_photons_proj :
    AxionLimits object for projected photon coupling limits.
limits_protons :
    AxionLimits object for proton coupling limits.
limits_protons_proj :
    AxionLimits object for projected proton coupling limits.
limits_neutrons :
    AxionLimits object for neutron coupling limits.
limits_neutrons_proj :
    AxionLimits object for projected neutron coupling limits.
limits_tops :
    AxionLimits object for top quark coupling limits.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
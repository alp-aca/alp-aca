"""alpaca.sectors

This module contains classes and functions to handle sectors of observables.

Classes
-------
Sector :
    A class representing a sector of observables.

Functions
---------
combine_sectors :
    Combine multiple sectors into a single sector.
initialize_sectors :
    Initialize sectors from a directory containing YAML files.

Objects
---------
default_sectors :
    Sectors initialized from the default directory.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
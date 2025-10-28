"""alpaca.scan

This module contains functions to scan over parameter space.

Classes
---------
Axis :
    Represents a parameter axis in the scan.

Scan :
    Represents a scan over parameter space.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
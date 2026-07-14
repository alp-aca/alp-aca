"""alpaca.export
    Export functions for ALP couplings to various formats.

Functions
---------
feynrules_export :
    Export ALP couplings to FeynRules format.
ufo_export :
    Export ALP couplings to UFO format.
"""


import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
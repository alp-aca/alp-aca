"""alpaca.benchmarks

This module contains the benchmarks defined in 1901.09966 [hep-ex] for ALPs.

Classes
---------
BC9 :
    Represents a photo-phillic ALP.

BC10 :
    Represents an ALP with universal couplings to fermions.

BC11 :
    Represents a gluon-phillic ALP.

Benchmark :
    Base class for all benchmarks.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
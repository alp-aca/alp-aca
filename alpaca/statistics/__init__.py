"""alpaca.statistics

This module contains functions to handle statistics.

Functions
---------
get_chi2 :
    Calculates the per-experiment chi-squared value for a list of processes.

combine_chi2 :
    Combines the chi-squared values of multiple experiments.

nsigmas :
    Calculates the number of standard deviations a value is from the mean.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
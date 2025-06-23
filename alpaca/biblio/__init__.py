"""alpaca.biblio

This module contains functions to handle bibliographical references.

Objects
-------
citations
    Keeps track of the citations used in the code globally.

Functions
---------
citations_context
    A context manager to retrieve citations from a block of code.

citation_report
    Generates a citation report for the measurements from a dict of InSpire ids.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
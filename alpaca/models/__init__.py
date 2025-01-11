"""alpaca.models

This module contains the classes to define the UV models.

Classes
-------
model :
    A class to define a model given the PQ charges of the SM fermions.

KSVZ_model :
    A class to define the KSVZ-like models given the new heavy fermions.

fermion :
    A class to represent a heavy fermion with specific group representations and charges.

Objects
-------
QED_DFSZ :
    A DFSZ-like model with couplings to leptons and quarks that does not generate a QCD anomaly.

u_DFSZ :
    A DFSZ-like model with couplings to leptons and up-type quarks.

d_DFSZ :
    A DFSZ-like model with couplings to leptons and down-type quarks.

Q_KSVZ :
    A KSVZ-like model with a heavy vector-like quark.

L_KSVZ :
    A KSVZ-like model with a heavy vector-like lepton.

beta :
    A symbol representing the angle beta in the DFSZ-like models.

KSVZ_charge :
    A symbol representing the charge of the heavy fermions in the KSVZ-like models.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
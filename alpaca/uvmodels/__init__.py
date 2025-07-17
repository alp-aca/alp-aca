"""alpaca.uvmodels

This module contains the classes to define the UV models.

Classes
-------
ModelBase :
    A base class to define a model which generates couplings to ALPs.

PQChargedModel :
    A class to define a model given the PQ charges of the SM fermions.

KSVZ_model :
    A class to define the KSVZ-like models given the new heavy fermions.

HeavyFermion :
    A class to represent a heavy fermion with specific group representations and charges.

Flaxion :
    A class to define a model with a flaxion field.

Objects
-------
QED_DFSZ :
    A DFSZ-like model with couplings to leptons and quarks that does not generate a QCD anomaly.

u_DFSZ :
    A DFSZ-like model where the up-type quarks are decoupled.

e_DFSZ :
    A DFSZ-like model where the leptons are decoupled.

Q_KSVZ :
    A KSVZ-like model with a heavy vector-like quark.

L_KSVZ :
    A KSVZ-like model with a heavy vector-like lepton charged under SU(2)_L.

Y_KSVZ :
    A KSVZ-like model with a heavy vector-like lepton charged under U(1)Y.

flaxion_benchmark :
    A benchmark model for the flaxion field.

nonuniversal_DFSZ :
    A DFSZ-like model with non-universal couplings to the SM fermions.

eps_flaxion :
    A symbol representing the parameter epsilon in the flaxion model.

beta :
    A symbol representing the angle beta in the DFSZ-like models.

KSVZ_charge :
    A symbol representing the charge of the heavy fermions in the KSVZ-like models.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
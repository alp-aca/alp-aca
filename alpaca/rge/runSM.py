import wilson
import ckmutil
import numpy as np
from cmath import phase
from ..biblio.biblio import citations
from functools import lru_cache
from ..common import svd, diagonalise_yukawas

def runSM(scale: float) -> dict:
    """SM parameters at an energy scale
    
    Parameters
    ----------
    scale : float
        Energy scale, in GeV

    Returns
    -------
    pars : dict
        dict containing the Yukawa matrices `yu`, `yd` and `ye`, the gauge couplings `alpha_s`, `alpha_1` and `alpha_2`, the sine squared of the Weinberg angle `s2w` and the CKM matrix `CKM`.
    """
    return _runSM(scale)

@lru_cache
def _runSM(scale):
    citations.register_inspire('Aebischer:2018bkb') #wilson
    citations.register_inspire('Straub:2018kue') # ckmutil is inside flavio's repo
    wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', scale, {})).C_in # For the moment we reuse wilson's code for the SM case, i.e, with all Wilson coefficients set to zero. Maybe at some point we should implement our own version.

    d_y = diagonalise_yukawas(wSM['Gu'], wSM['Gd'], wSM['Ge'])
    UuL, mu, UuR = d_y['u']
    UdL, md, UdR = d_y['d']
    Vckm = UuL.conj().T @ UdL

    return {
        'yu': np.matrix(UuL).H @ np.matrix(wSM['Gu']),
        'yd': np.matrix(UuL).H @ np.matrix(wSM['Gd']),
        'ye': np.matrix(wSM['Ge']),
        'alpha_s': np.real(wSM['gs']**2/(4*np.pi)),
        'alpha_1': np.real(wSM['gp']**2/(4*np.pi)),
        'alpha_2': np.real(wSM['g']**2/(4*np.pi)),
        'alpha_em': np.real(wSM['g']**2*wSM['gp']**2/(wSM['g']**2+wSM['gp']**2))/(4*np.pi),
        's2w': np.real(wSM['gp']**2/(wSM['g']**2+wSM['gp']**2)),
        'CKM': np.matrix(Vckm),
        'vev': np.sqrt(2*wSM['m2']/wSM['Lambda']),
        'GF': 1/(2*wSM['m2']/(wSM['Lambda']))/2**0.5
    }
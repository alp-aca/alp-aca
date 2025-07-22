import wilson
import ckmutil
import numpy as np
from cmath import phase
from ..biblio.biblio import citations
from functools import lru_cache

def svd(A):
    u, s, vh = np.linalg.svd(A)
    pmatrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    u = u @ pmatrix
    s = s[[2,1,0]]
    vh = pmatrix @ vh
    t = np.eye(3)
    if np.real(u[0,0]) < 0:
        t[0,0] = -1
    if np.real(u[1,1]) < 0:
        t[1,1] = -1
    if np.real(u[2,2]) < 0:
        t[2,2] = -1
    return u @ t, s, t @ vh

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

    UuL, mu, UuR = svd(wSM['Gu'])
    UdL, md, UdR = svd(wSM['Gd'])
    K = UuL.conj().T @ UdL
    Vub = abs(K[0,2])
    Vcb = abs(K[1,2])
    Vus = abs(K[0,1])
    gamma = phase(-K[0,0]*K[0,2].conj()/(K[1,0]*K[1,2].conj()))
    Vckm = ckmutil.ckm.ckm_tree(Vus, Vub, Vcb, gamma)

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
import wilson
import ckmutil
import numpy as np
from cmath import phase

def runSM(scale):
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

    wSM = wilson.classes.SMEFT(wilson.wcxf.WC('SMEFT', 'Warsaw', scale, {})).C_in # For the moment we reuse wilson's code for the SM case, i.e, with all Wilson coefficients set to zero. Maybe at some point we should implement our own version.

    UuL, mu, UuR = ckmutil.diag.msvd(wSM['Gu'])
    UdL, md, UdR = ckmutil.diag.msvd(wSM['Gd'])
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
        'CKM': np.matrix(Vckm)
    }
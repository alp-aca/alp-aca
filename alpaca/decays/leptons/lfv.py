from ...rge import ALPcouplings
from ...common import kallen
from ...constants import me, mmu, mtau
import numpy as np

def lfv_decay(ma: float, ml1: float, ml2: float, cV: complex, cA: complex, fa: float):
    amp_sq = (np.abs(cA)**2 * (ml1+ml2)**2 * ((ml1-ml2)**2-ma**2) + np.abs(cV)**2 * (ml1-ml2)**2 * ((ml1+ml2)**2-ma**2) ) / (4*fa**2)
    return amp_sq*np.sqrt(kallen(ma**2, ml1**2, ml2**2)) / (16*np.pi*ml1**3)

def tau_mua(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    if ma > mtau - mmu:
        return 0
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    cV = cc['ceV'][1,2]
    cA = cc['ceA'][1,2]
    return lfv_decay(ma, mtau, mmu, cV, cA, fa)

def tau_ea(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    if ma > mtau - me:
        return 0
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    cV = cc['ceV'][0,2]
    cA = cc['ceA'][0,2]
    return lfv_decay(ma, mtau, me, cV, cA, fa)

def mu_ea(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    if ma > mmu - me:
        return 0
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    cV = cc['ceV'][0,1]
    cA = cc['ceA'][0,1]
    return lfv_decay(ma, mmu, me, cV, cA, fa)
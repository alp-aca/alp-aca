import numpy as np
from ...rge import ALPcouplings, bases_above
from ...constants import me, mmu, mtau, mc, mb
from ...biblio.biblio import citations
from ..effcouplings import effcoupling_ff


def fermion_decay_width(ma, fa,cf, mf,Nc):
    citations.register_inspire('Bauer:2017ris')
    if mf<ma/2:
        return (Nc*np.abs(cf)**2/fa**2)*np.sqrt(1-pow(2*mf/ma,2))*ma*pow(mf,2)/(8 * np.pi)
    else:
        return 0.0

def decay_width_electron(ma, couplings: ALPcouplings,fa,**kwargs):
    ceA = effcoupling_ff(ma, couplings, 'e', **kwargs)
    return fermion_decay_width(ma, fa, ceA, me, Nc=1)

def decay_width_muon(ma,couplings: ALPcouplings,fa, **kwargs):
    ceA = effcoupling_ff(ma, couplings, 'mu', **kwargs)
    return fermion_decay_width(ma, fa, ceA, mmu, Nc=1)

def decay_width_tau(ma,couplings: ALPcouplings,fa,**kwargs):
    ceA = effcoupling_ff(ma, couplings, 'tau', **kwargs)
    return fermion_decay_width(ma, fa, ceA, mtau, Nc=1)

def decay_width_charm(ma,couplings: ALPcouplings,fa,**kwargs):
    cuA = effcoupling_ff(ma, couplings, 'c', **kwargs)
    return fermion_decay_width(ma, fa, cuA, mc, Nc=3)

def decay_width_bottom(ma,couplings: ALPcouplings,fa,**kwargs):
    cdA = effcoupling_ff(ma, couplings, 'b', **kwargs)
    return fermion_decay_width(ma, fa, cdA, mb, Nc=3)

def dw_lfv(ma: float, fa: float, cV: complex, cA: complex, ml1: float, ml2: float):
    if ma < ml1 + ml2:
        return 0.0
    citations.register_inspire('Calibbi:2020jvd')
    zp = 1.0 - (ml1+ml2)**2/ma**2
    zm = 1.0 - (ml1-ml2)**2/ma**2
    return ma/(32*np.pi*fa**2)*(np.abs(cV)**2 *(ml1-ml2)**2*zp + np.abs(cA)**2*(ml1+ml2)**2*zm)*np.sqrt(zp*zm)

def decay_width_mue(ma, couplings: ALPcouplings, fa,**kwargs):
    if ma < couplings.ew_scale:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
        cV = cc['ceV'][0,1]
        cA = cc['ceA'][0,1]
    else:
        cc = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        cV = cc['ceL'][0,1]-cc['ceL'][1,0]
        cA = cc['ceL'][0,1]+cc['ceL'][1,0]
    return dw_lfv(ma, fa, cV, cA, me, mmu)

def decay_width_mutau(ma, couplings: ALPcouplings, fa,**kwargs):
    if ma < couplings.ew_scale:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
        cV = cc['ceV'][1,2]
        cA = cc['ceA'][1,2]
    else:
        cc = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        cV = cc['ceL'][1,2]-cc['ceL'][2,1]
        cA = cc['ceL'][1,2]+cc['ceL'][2,1]
    return dw_lfv(ma, fa, cV, cA, mmu, mtau)

def decay_width_etau(ma, couplings: ALPcouplings, fa,**kwargs):
    if ma < couplings.ew_scale:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
        cV = cc['ceV'][0,2]
        cA = cc['ceA'][0,2]
    else:
        cc = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        cV = cc['ceL'][0,2]-cc['ceL'][0,2]
        cA = cc['ceL'][0,2]+cc['ceL'][0,2]
    return dw_lfv(ma, fa, cV, cA, mtau, me)
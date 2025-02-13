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




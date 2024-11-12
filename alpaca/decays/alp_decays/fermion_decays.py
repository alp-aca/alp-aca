import numpy as np
from ...rge import ALPcouplings, bases_above
from ...constants import me, mmu, mtau, mc, mb
from ...citations import citations
from ..effective_couplings import clepton


def fermion_decay_width(ma, fa,cf, mf,Nc):
    citations.register_inspire('Bauer:2017ris')
    if mf<ma/2:
        return (Nc*np.abs(cf)**2/fa**2)*np.sqrt(1-pow(2*mf/ma,2))*ma*pow(mf,2)/(8 * np.pi)
    else:
        return 0.0

def decay_width_electron(ma, couplings: ALPcouplings,fa,**kwargs):
    matching_scale = kwargs.get('matching_scale', 100)
    if ma > matching_scale:
        cc = couplings.match_run(ma, 'massbasis_above', **kwargs)
    else:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
    return fermion_decay_width(ma, fa, clepton(cc)[0,0],me,Nc=1)

def decay_width_muon(ma,couplings: ALPcouplings,fa, **kwargs):
    matching_scale = kwargs.get('matching_scale', 100)
    if ma > matching_scale:
        cc = couplings.match_run(ma, 'massbasis_above', **kwargs)
    else:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
    return fermion_decay_width(ma, fa, clepton(cc)[1,1], mmu, Nc=1)

def decay_width_tau(ma,couplings: ALPcouplings,fa,**kwargs):
    matching_scale = kwargs.get('matching_scale', 100)
    if ma > matching_scale:
        cc = couplings.match_run(ma, 'massbasis_above', **kwargs)
    else:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
    return fermion_decay_width(ma, fa, clepton(cc)[2,2], mtau, Nc=1)

def decay_width_charm(ma,couplings: ALPcouplings,fa,**kwargs):
    matching_scale = kwargs.get('matching_scale', 100)
    if ma > matching_scale:
        cc = couplings.match_run(ma, 'massbasis_above', **kwargs)
        cuA = cc['ku'] - cc['kU']
    else:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
        cuA = cc['cuA']
    return fermion_decay_width(ma, fa, cuA[1,1], mc, Nc=3)

def decay_width_bottom(ma,couplings: ALPcouplings,fa,**kwargs):
    matching_scale = kwargs.get('matching_scale', 100)
    if ma > matching_scale:
        cc = couplings.match_run(ma, 'massbasis_above', **kwargs)
        cdA = cc['kd'] - cc['kD']
    else:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
        cdA = cc['cdA']
    return fermion_decay_width(ma, fa, cdA[2,2], mb, Nc=3)




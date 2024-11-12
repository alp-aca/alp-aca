from ...rge import ALPcouplings
from ...common import alpha_s, alpha_em
from ...citations import citations
from ..effective_couplings import cgamma, cg
import numpy as np


def decay_width_2gamma(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    citations.register_inspire('Bauer:2017ris')
    if ma > kwargs.get('matching_scale', 100):
        basis = 'massbasis_above'
    else:
        basis = 'VA_below'
    return alpha_em(ma)**2*ma**3*np.abs(cgamma(couplings.match_run(ma, basis, **kwargs), fa, **kwargs))**2/((4*np.pi)**3*fa**2)

def decay_width_2gluons(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    citations.register_inspire('Bauer:2017ris')
    if ma < 1.84:
        return 0.0
    if ma > kwargs.get('matching_scale', 100):
        basis = 'massbasis_above'
    else:
        basis = 'VA_below'
    
    return alpha_s(ma)**2*ma**3/((4*np.pi)**3*fa**2)*np.abs(cg(couplings.match_run(ma, basis, **kwargs)))**2*(1+alpha_s(ma)/np.pi*83/4)#(1+alpha_s(ma)/48/np.pi*(291-sum(14 for i in range(5) if ma > mq[i])))
from ..effcouplings import effcoupling_gammaZ
from ...constants import mZ, s2w
from ...common import alpha_em
from ...biblio.biblio import citations
from ...rge import ALPcouplings
import numpy as np

def decaywidth_Z_to_agamma(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    if ma > mZ:
        return 0
    citations.register_inspire('Bonilla:2021ufe')
    ceff = effcoupling_gammaZ(couplings, ma)
    c2w = 1-s2w
    return (alpha_em(ma) * mZ**3/(96*np.pi**3*fa**2*s2w*c2w))*(1 - ma**2/mZ**2)**3 * np.abs(ceff)**2
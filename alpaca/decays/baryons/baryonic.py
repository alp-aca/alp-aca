from ..effcouplings import effcoupling_baryons_A, effcoupling_baryons_V
import numpy as np
from ...rge import ALPcouplings
from ...constants import mproton, mneutron, mSigma0, mSigma_minus, mSigma_plus, mLambda, mXi0, mXi_minus
from ...common import kallen
from ...constants import GammaSigma0, GammaSigma_plus, GammaSigma_minus, GammaLambda, GammaXi0, GammaXi_minus
from ..nwa import transition_nwa
from ..alp_decays.branching_ratios import decay_channels


_masses = {
    'proton': mproton,
    'neutron': mneutron,
    'Sigma0': mSigma0,
    'Sigma+': mSigma_plus,
    'Sigma-': mSigma_minus,
    'Lambda': mLambda,
    'Xi0': mXi0,
    'Xi-': mXi_minus
}

def decay_width_prod(ma: float, couplings: ALPcouplings, fa: float, b1: str, b2: str, **kwargs):
    # b1 -> b2 a
    m1 = _masses[b1]
    m2 = _masses[b2]
    if m1 < ma + m2:
        return 0.0
    cA = effcoupling_baryons_A(couplings, ma, b1, b2, **kwargs)
    cV = effcoupling_baryons_V(couplings, ma, b1, b2, **kwargs)
    amp_sq = (np.abs(cA)**2 * (m1+m2)**2 * ((m1-m2)**2-ma**2) + np.abs(cV)**2 * (m1-m2)**2 * ((m1+m2)**2-ma**2) ) / (4*fa**2)
    return amp_sq*np.sqrt(kallen(ma**2, m1**2, m2**2)) / (16*np.pi*m1**3)

baryon_to_alp = {
    ('Sigma+', ('alp', 'proton')): lambda ma, couplings, fa, br_dark, **kwargs: decay_width_prod(ma, couplings, fa, 'Sigma+', 'proton', **kwargs)/GammaSigma_plus,
    ('Sigma0', ('alp', 'neutron')): lambda ma, couplings, fa, br_dark, **kwargs: decay_width_prod(ma, couplings, fa, 'Sigma0', 'neutron', **kwargs)/GammaSigma0,
    ('Lambda', ('alp', 'neutron')): lambda ma, couplings, fa, br_dark, **kwargs: decay_width_prod(ma, couplings, fa, 'Lambda', 'neutron', **kwargs)/GammaLambda,
    ('Xi0', ('alp', 'Sigma0')): lambda ma, couplings, fa, br_dark, **kwargs: decay_width_prod(ma, couplings, fa, 'Xi0', 'Sigma0', **kwargs)/GammaXi0,
    ('Xi-', ('alp', 'Sigma-')): lambda ma, couplings, fa, br_dark, **kwargs: decay_width_prod(ma, couplings, fa, 'Xi-', 'Sigma-', **kwargs)/GammaXi_minus,
    ('Xi0', ('alp', 'Lambda')): lambda ma, couplings, fa, br_dark, **kwargs: decay_width_prod(ma, couplings, fa, 'Xi0', 'Lambda', **kwargs)/GammaXi0,
    ('Sigma0', ('alp', 'Lambda')): lambda ma, couplings, fa, br_dark, **kwargs: decay_width_prod(ma, couplings, fa, 'Sigma0', 'Lambda', **kwargs)/GammaSigma0,
}

baryon_nwa = {}
for baryon_process in baryon_to_alp.keys():
    for channel in decay_channels:
        baryon_nwa[transition_nwa(baryon_process, channel)] = (baryon_process, channel)
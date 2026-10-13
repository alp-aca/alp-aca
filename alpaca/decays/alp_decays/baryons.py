import numpy as np
from ...biblio.biblio import citations
from ..effcouplings import effcoupling_baryons_A, effcoupling_baryons_V
from ...rge import ALPcouplings
from ...constants import mproton, mneutron, mSigma0, mSigma_minus, mSigma_plus, mLambda, mXi0, mXi_minus

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

def dw_fv(ma: float, fa: float, cV: complex, cA: complex, m1: float, m2: float):
    if ma < m1 + m2:
        return 0.0
    citations.register_inspire('Calibbi:2020jvd')
    zp = 1.0 - (m1+m2)**2/ma**2
    zm = 1.0 - (m1-m2)**2/ma**2
    return ma/(32*np.pi*fa**2)*(np.abs(cV)**2 *(m1-m2)**2*zp + np.abs(cA)**2*(m1+m2)**2*zm)*np.sqrt(zp*zm)

def dw_baryons(ma: float, couplings: ALPcouplings, fa: float, b1: str, b2: str, **kwargs):
    cA = 2 * effcoupling_baryons_A(couplings, ma, b1, b2, **kwargs)
    cV = 2 * effcoupling_baryons_V(couplings, ma, b1, b2, **kwargs)
    if b1 == b2:
        conjugate_decay = 1
    else:
        conjugate_decay = 2
    return conjugate_decay * dw_fv(ma, fa, cV, cA, _masses[b1], _masses[b2])

channels_baryons = [
    ('proton', 'proton'),
    ('neutron', 'neutron'),
    ('Lambda', 'Lambda'),
    ('Sigma+', 'Sigma+'),
    ('Sigma0', 'Sigma0'),
    ('Sigma-', 'Sigma-'),
    ('Xi0', 'Xi0'),
    ('Xi-', 'Xi-'),
    ('Lambda', 'Sigma0'),
    ('Sigma+', 'proton'),
    ('Sigma0', 'neutron'),
    ('Lambda', 'neutron'),
    ('Sigma0', 'Xi0'),
    ('Sigma-', 'Xi-'),
    ('Lambda', 'Xi0'),
]
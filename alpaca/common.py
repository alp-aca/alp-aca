import numpy as np
import flavio
from .constants import pars
from pyCollier import c0
from .citations import citations


def kallen(a, b, c):
    return a**2+b**2+c**2-2*a*b-2*a*c-2*b*c


def floop(x):
    if x >= 1:
        return np.arcsin(x**(-0.5))
    else:
        return np.pi/2+0.5j*np.log((1+np.sqrt(1-x))/(1-np.sqrt(1-x)))

B1 = lambda x: 1-x*floop(x)**2
B2 = lambda x: 1-(x-1)*floop(x)**2


alpha_em = lambda q: flavio.physics.running.running.get_alpha_e(pars, q)
alpha_s = lambda q: flavio.physics.running.running.get_alpha_s(pars, q)

f0_BK = lambda q2: flavio.physics.bdecays.formfactors.b_p.bcl.ff('B->K', q2, pars)['f0']
f0_Kpi = lambda q2: flavio.physics.kdecays.formfactors.fp0_dispersive(q2, pars)['f0']
A0_BKst = lambda q2: flavio.physics.bdecays.formfactors.b_v.bsz.ff('B->K*', q2, pars)['A0']

def scalarC0(s1: float | np.ndarray, s2: float | np.ndarray, s12: float | np.ndarray, m0: float | np.ndarray, m1: float | np.ndarray, m2: float | np.ndarray) -> complex | np.ndarray:
    """
    Compute the scalar Passarino-Veltman C0 function, using the same definition as Package-X. Also works with NumPy arrays.

    Parameters:
        s1 (float): Invariant mass squared of the first external particle, s1 = (p1)**2.
        s2 (float): Invairant mass squared of the second external particle, s2 = (p2)**2.
        s12 (float): Invariant mass squared of the third external particle, s12 = (p1+p2)**2 = (p3)**2.
        m0 (float): Mass of the internal particle between the p1 and p2 legs.
        m1 (float): Mass of the internal particle between the p2 and p3 legs.
        m2 (float): Mass of the internal particle between the p3 and p1 legs.

    Returns:
        complex: The result of the scalar C0 function.
    """

    citations.register_inspire('Passarino:1978jh') # Passarino-Veltman paper
    citations.register_inspire('tHooft:1978jhc') # 't Hooft-Veltman paper
    citations.register_inspire('Denner:2016kdg') # collier paper

    return np.vectorize(c0)(s1, s2, s12, m0**2, m1**2, m2**2)

def discB0(s: float, m1: float, m2: float) -> complex:
    """
    Compute the part of the Passarino-Veltman B0 function containing the s-plane branch cut, using the same definition as Package-X.

    Parameters:
        s (float): Invariant mass squared of the external particle, s = (p)**2.
        m1 (float): Mass of one of the internal particles.
        m2 (float): Mass of the  other internal particle.

    Returns:
        complex: The result of the discontinuity of the scalar B0 function.
    """

    citations.register_inspire('Passarino:1978jh') # Passarino-Veltman paper
    citations.register_inspire('tHooft:1978jhc') # 't Hooft-Veltman paper

    eps = (1-1e-10j) # Small imaginary part to avoid branch cut
    return np.sqrt(kallen(s, m1**2, m2**2)*eps)/s * np.log((m1**2+m2**2-s+np.sqrt(kallen(s, m1**2, m2**2)*eps))/(2*m1*m2)*eps)
import numpy as np
import flavio
from .constants import pars
from .biblio.biblio import citations
from scipy.integrate import quad
import warnings
from functools import cache

def kallen(a, b, c):
    return a**2+b**2+c**2-2*a*b-2*a*c-2*b*c


def floop(x):
    if x >= 1:
        return np.arcsin(x**(-0.5))
    else:
        return np.pi/2+0.5j*np.log((1+np.sqrt(1-x))/(1-np.sqrt(1-x)))

B1 = lambda x: 1-x*floop(x)**2
B2 = lambda x: 1-(x-1)*floop(x)**2
B3 = lambda x, y: 1 + x*y/(x-y)*(floop(x)**2 - floop(y)**2)

def B0disc_equalmass(q2: float, m: float) -> complex:
    return 2j*np.sqrt(1+0j-4*m**2/q2)*floop(np.sqrt(4*m**2/q2))

@cache
def g_photonloop(tau):
    def int_g(x, tau):
        tau *= (1-1e-8j)
        den = np.sqrt(tau*(1-x)**2-x**2)
        atan = np.arctan(x/den)
        num = 1-4*tau*(1-x)**2-2*x+4*x**2
        return num/den*atan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return quad(int_g, 0, 1, args=tau, complex_func=True)[0]/3*4+5

alpha_em = lambda q: flavio.physics.running.running.get_alpha_e(pars, q)
alpha_s = lambda q: flavio.physics.running.running.get_alpha_s(pars, q)

def f0_BK(q2):
    citations.register_inspire('FlavourLatticeAveragingGroupFLAG:2021npn')
    return flavio.physics.bdecays.formfactors.b_p.bcl.ff('B->K', q2, pars)['f0']

def f0_Bpi(q2):
    citations.register_inspire('Leljak:2021vte')
    return flavio.physics.bdecays.formfactors.b_p.bcl_lmvd.ff('B->pi', q2, pars)['f0']

def f0_Kpi(q2):
    citations.register_inspire('FlaviaNetWorkingGrouponKaonDecays:2010lot')
    citations.register_inspire('Antonelli:2010yf')
    citations.register_inspire('FlavourLatticeAveragingGroupFLAG:2021npn')
    return flavio.physics.kdecays.formfactors.fp0_dispersive(q2, pars)['f0']

def A0_BKst(q2):
    citations.register_inspire('Horgan:2015vla')
    return flavio.physics.bdecays.formfactors.b_v.bsz.ff('B->K*', q2, pars)['A0']

def A0_Brho(q2):
    citations.register_inspire('Bharucha:2015bzk')
    return flavio.physics.bdecays.formfactors.b_v.bsz.ff('B->rho', q2, pars)['A0']

def A0_Bsphi(q2):
    citations.register_inspire('Bharucha:2015bzk')
    return flavio.physics.bdecays.formfactors.b_v.bsz.ff('Bs->phi', q2, pars)['A0']

ckm_xi = lambda i, j: flavio.physics.ckm.xi(i, j)(pars)

def f0_Dpi(q2):
    citations.register_inspire('Lubicz:2017syv')
    return flavio.physics.ddecays.formfactors.bsz.ff('D->pi', q2, pars)['f0']

def f0_DsK(q2):
    citations.register_inspire('Wang:2008ci')
    f0_0 = 0.67
    a = 0.50
    b = -0.005
    from .constants import mDs
    return f0_0/(1-a*q2/mDs+b*q2**2/mDs**2)

def f0_Deta(q2):
    citations.register_inspire('Palmer:2013yia')
    citations.register_inspire('Fajfer:2004mv')
    f0_0 = 0.66
    mDprime = 2.3
    mDst = 2.01
    return f0_0/(1-q2*mDprime**2/mDst**4)

def f0_Detap(q2):
    citations.register_inspire('Palmer:2013yia')
    citations.register_inspire('Fajfer:2004mv')
    f0_0 = 0.55
    mDprime = 2.3
    mDst = 2.01
    return f0_0/(1-q2*mDprime**2/mDst**4)

def A0_Drho(q2):
    citations.register_inspire('Chang:2019mmh')
    A0_0 = 0.68
    a = 1.27
    b = 0.30
    from .constants import mDs
    return A0_0/(1-a*q2/mDs+b*q2**2/mDs**2)

def A0_DsKst(q2):
    citations.register_inspire('Chang:2019mmh')
    A0_0 = 0.76
    a = 1.14
    b = 0.26
    from .constants import mDs
    return A0_0/(1-a*q2/mDs+b*q2**2/mDs**2)
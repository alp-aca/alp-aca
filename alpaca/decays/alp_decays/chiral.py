from ...rge import ALPcouplings 
import numpy as np
from ...constants import mu, md, ms, mpi0, meta, metap, fpi
from ...common import alpha_s
from . import u3reprs
from ...biblio.biblio import citations
from functools import lru_cache
import pickle
import os


#with open(os.path.join(path, 'ffunction.pickle'), 'rb') as f:
#    ffunction_interp = pickle.load(f)

kappa = np.diag([1/m for m in [mu, md, ms]])/sum(1/m for m in [mu, md, ms])

def kinetic_mixing(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.ndarray:
    citations.register_inspire('Aloni:2018vki')
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    cq_eff = np.array([cc['cuA'][0,0], cc['cdA'][0,0], cc['cdA'][1,1]]) - 2*cc['cg']*kappa
    eps = fpi/fa
    return np.array([eps/4*(cq_eff[0,0]-cq_eff[1,1]), eps/2/np.sqrt(6)*(cq_eff[0,0]+cq_eff[1,1]-cq_eff[2,2]), eps/4/np.sqrt(3)*(cq_eff[0,0]+cq_eff[1,1]+2*cq_eff[2,2])])

def mass_mixing(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.ndarray:
    citations.register_inspire('Aloni:2018vki')
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    m0 = mpi0**2/(mu+md)*mu*md*ms/(mu*md+mu*ms+md*ms)
    eps = fpi/fa
    return np.array([0, -cc['cg']*eps*np.sqrt(2/3)*m0, -cc['cg']*eps*np.sqrt(2/3)*m0*2*np.sqrt(2)])

def a_U3_proj(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.ndarray:
    citations.register_inspire('Aloni:2018vki')
    m_mesons = np.array([mpi0, meta, metap])
    return (mass_mixing(ma, couplings, fa, **kwargs)-ma**2*kinetic_mixing(ma, couplings, fa, **kwargs))/(ma**2-m_mesons**2)

@lru_cache
def a_U3_repr(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.matrix:
    citations.register_inspire('Aloni:2018vki')
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    coup_q = ALPcouplings({'cuA': cc['cuA'], 'cdA': cc['cdA']}, ma, 'VA_below')
    coup_g = ALPcouplings({'cg': cc['cg']}, ma, 'VA_below')
    components_q = a_U3_proj(ma, coup_q, fa, **kwargs)
    components_g = a_U3_proj(ma, coup_g, fa, **kwargs)
    u3repr_q = components_q[0]*u3reprs.pi0 + components_q[1]*u3reprs.eta + components_q[2]*u3reprs.etap
    u3repr_g = components_g[0]*u3reprs.pi0 + components_g[1]*u3reprs.eta + components_g[2]*u3reprs.etap
    if ma > 1.0:
        u3repr_g[0,0] = u3repr_g[1,1]
    if ma > 1.15:
        u3repr_g[0,0] = u3repr_g[1,1] = u3repr_g[2,2] = cc['cg']* fpi/fa*alphas_tilde(ma)/np.sqrt(6)
    return u3repr_q + u3repr_g

def alphas_tilde(ma: float) -> float:
    if ma < 1.0:
        return 1.0
    if ma < 1.5:
        return 2*ma*(alpha_s(1.5)-1)+3-2*alpha_s(1.5)
    return alpha_s(ma)

class _ffunction:
    ffunction_interp = None
    initialized = False

    def initialize(self):
        path = os.path.dirname(__file__)
        with open(os.path.join(path, 'ffunction.pickle'), 'rb') as f:
            self.ffunction_interp = pickle.load(f)
        self.initialized = True

    def __call__(self, ma):
        #INPUT:
            #ma: Mass of ALP (GeV)
        #OUTPUT:
            #Data-driven function 
        #Chiral contribution (1811.03474, eq. S26, ,approx) (below mass eta')

        if not self.initialized:
            self.initialize()
        citations.register_inspire('Aloni:2018vki')
        citations.register_inspire('BaBar:2017zmc')
        citations.register_inspire('BaBar:2007ceh')
        citations.register_inspire('BaBar:2004ytv')
        if ma < 1.4: fun = 1
        elif ma >= 1.4 and ma <= 2: 
            fun = self.ffunction_interp(ma)
        else: fun = (1.4/ma)**4
        return fun
    
ffunction = _ffunction()
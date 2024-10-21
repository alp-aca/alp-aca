from ...rge import ALPcouplings 
import numpy as np
from ...constants import mu, md, ms, mpi0, meta, metap, fpi
from ...common import alpha_s
from . import u3reprs

kappa = np.diag([1/m for m in [mu, md, ms]])/sum(1/m for m in [mu, md, ms])

def kinetic_mixing(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.ndarray:
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    cq_eff = np.array([cc['cuA'][0,0], cc['cdA'][0,0], cc['cdA'][1,1]]) - 2*cc['cg']*kappa
    eps = fpi/fa
    return np.array([-eps/4*(cq_eff[0,0]-cq_eff[1,1]), -eps/2/np.sqrt(6)*(cq_eff[0,0]+cq_eff[1,1]-cq_eff[2,2]), -eps/4/np.sqrt(3)*(cq_eff[0,0]+cq_eff[1,1]+2*cq_eff[2,2])])

def mass_mixing(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.ndarray:
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    m0 = mpi0**2/(mu+md)*mu*md*ms/(mu*md+mu*ms+md*ms)
    eps = fpi/fa
    return np.array([0, -cc['cg']*eps*np.sqrt(2/3)*m0, -cc['cg']*eps*np.sqrt(2/3)*m0*2*np.sqrt(2)])

def a_U3_proj(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.ndarray:
    m_mesons = np.array([mpi0, meta, metap])
    return (mass_mixing(ma, couplings, fa, **kwargs)-ma**2*kinetic_mixing(ma, couplings, fa, **kwargs))/(ma**2-m_mesons**2)

def a_U3_repr(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.matrix:
    #if ma > metap:
    #    raise NotImplementedError
    components = a_U3_proj(ma, couplings, fa, **kwargs)
    return components[0]*u3reprs.pi0 + components[1]*u3reprs.eta + components[2]*u3reprs.etap

def alphas_tilde(ma: float) -> float:
    if ma < 1.0:
        return 1.0
    if ma < 1.5:
        return 2*ma*(alpha_s(1.5)-1)+3-2*alpha_s(1.5)
    return alpha_s(ma)
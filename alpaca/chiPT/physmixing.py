import numpy as np
from .chiral import sm_mixingangles, a_U3_repr, kappa
from ..rge import ALPcouplings
from .u3reprs import pi0, eta, etap, K0, K0bar
from ..constants import mu, md, theta_eta_etap, fpi
from ..biblio import citations

def mixing_K0_alp(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> complex:
    return 2*np.trace(a_U3_repr(ma, couplings, fa, **kwargs) @ K0)

def mixing_K0bar_alp(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> complex:
    return 2*np.trace(a_U3_repr(ma, couplings, fa, **kwargs) @ K0bar)

def mixing_pi0_alp(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> complex:
    citations.register_inspire('Kyselov:2025uez')
    a = a_U3_repr(ma, couplings, fa, **kwargs)
    deltaI = (md-mu)/(md+mu)
    th_pi_a = 2*np.trace(a @ pi0)
    th_eta_a = 2*np.trace(a @ eta) *(np.cos(theta_eta_etap)-np.sqrt(2)*np.sin(theta_eta_etap))/np.sqrt(2)
    th_etap_a = 2*np.trace(a @ etap) *(np.sin(theta_eta_etap)+np.sqrt(2)*np.cos(theta_eta_etap))
    th_pi_eta = sm_mixingangles()[('pi0', 'eta')]*(np.cos(theta_eta_etap)-np.sqrt(2)*np.sin(theta_eta_etap))/np.sqrt(2)
    th_pi_etap = sm_mixingangles()[('pi0', 'etap')]*(np.sin(theta_eta_etap)+np.sqrt(2)*np.cos(theta_eta_etap))
    F0 = fpi/np.sqrt(2)
    cG = couplings['cG']*F0/fa

    return th_pi_a + cG*(kappa[0,0]-kappa[1,1]) - deltaI*(np.sqrt(3)*th_pi_etap + np.sqrt(6)*th_pi_eta + 1)*(cG*(kappa[0,0]+kappa[1,1]) + (th_etap_a + np.sqrt(2)*th_eta_a)/np.sqrt(3) )

def mixing_eta_alp(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> complex:
    citations.register_inspire('Kyselov:2025uez')
    a = a_U3_repr(ma, couplings, fa, **kwargs)
    deltaI = (md-mu)/(md+mu)
    th_pi_a = 2*np.trace(a @ pi0)
    th_eta_a = 2*np.trace(a @ eta) *(np.cos(theta_eta_etap)-np.sqrt(2)*np.sin(theta_eta_etap))/np.sqrt(2)
    th_etap_a = 2*np.trace(a @ etap) *(np.sin(theta_eta_etap)+np.sqrt(2)*np.cos(theta_eta_etap))
    th_pi_eta = sm_mixingangles()[('pi0', 'eta')]*(np.cos(theta_eta_etap)-np.sqrt(2)*np.sin(theta_eta_etap))/np.sqrt(2)
    th_pi_etap = sm_mixingangles()[('pi0', 'etap')]*(np.sin(theta_eta_etap)+np.sqrt(2)*np.cos(theta_eta_etap))
    F0 = fpi/np.sqrt(2)
    cG = couplings['cG']*F0/fa

    return th_eta_a + th_etap_a/np.sqrt(2) + np.sqrt(3/2)*cG*(kappa[0,0]+kappa[1,1])-0.5*deltaI*(cG*(kappa[0,0]-kappa[1,1])+th_pi_a)*(2*np.sqrt(2)*th_pi_etap + th_pi_eta+np.sqrt(6))

def mixing_etap_alp(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> complex:
    citations.register_inspire('Kyselov:2025uez')
    a = a_U3_repr(ma, couplings, fa, **kwargs)
    deltaI = (md-mu)/(md+mu)
    th_pi_a = 2*np.trace(a @ pi0)
    th_eta_a = 2*np.trace(a @ eta) *(np.cos(theta_eta_etap)-np.sqrt(2)*np.sin(theta_eta_etap))/np.sqrt(2)
    th_etap_a = 2*np.trace(a @ etap) *(np.sin(theta_eta_etap)+np.sqrt(2)*np.cos(theta_eta_etap))
    th_pi_eta = sm_mixingangles()[('pi0', 'eta')]*(np.cos(theta_eta_etap)-np.sqrt(2)*np.sin(theta_eta_etap))/np.sqrt(2)
    th_pi_etap = sm_mixingangles()[('pi0', 'etap')]*(np.sin(theta_eta_etap)+np.sqrt(2)*np.cos(theta_eta_etap))
    F0 = fpi/np.sqrt(2)
    cG = couplings['cG']*F0/fa

    return th_etap_a + np.sqrt(2)*th_eta_a + np.sqrt(3)*cG*(kappa[0,0]+kappa[1,1]) + deltaI*(cG*(kappa[0,0]-kappa[1,1])+th_pi_a)*(th_pi_etap - 2*np.sqrt(2)*th_pi_eta - np.sqrt(3))
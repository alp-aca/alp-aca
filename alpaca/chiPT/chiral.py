from ..rge import ALPcouplings 
import numpy as np
from ..constants import mu, md, ms, mpi0, fpi, theta_eta_etap
from ..common import alpha_s
from . import u3reprs
from ..biblio.biblio import citations
from functools import lru_cache
import pickle
import os


#with open(os.path.join(path, 'ffunction.pickle'), 'rb') as f:
#    ffunction_interp = pickle.load(f)

kappa = np.diag([1/m for m in [mu, md, ms]])/sum(1/m for m in [mu, md, ms])
#kappa = np.diag([0,1,0])

def cqhat(couplings: ALPcouplings, ma: float, **kwargs) -> np.ndarray:
    cc = couplings.match_run(ma, 'VA_below', **kwargs)
    cq = np.array([[cc['cuA'][0,0], 0, 0], [0, cc['cdA'][0,0], cc['cdA'][0,1]], [0, cc['cdA'][1,0], cc['cdA'][1,1]]])
    return cq + 2 * cc['cG'] * kappa

def mesonmass_chiPT():
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    return {
        'pi0': mpi0,
        'eta': mpi0/np.sqrt(3*c_eta) * np.sqrt((c_eta-np.sqrt(2)*s_eta) +2*ms/(mu+md)*(2*c_eta+np.sqrt(2)*s_eta)),
        'etap': mpi0*np.sqrt(((s_eta+np.sqrt(2)*c_eta) +2*ms/(mu+md)*(2*s_eta-np.sqrt(2)*c_eta))/3/s_eta),
        'K0': mpi0 * np.sqrt((md+ms)/(md+mu)),
        'K0bar': mpi0 * np.sqrt((md+ms)/(md+mu)),
    }

def kinetic_mixing(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> dict[str, complex]:
    citations.register_inspire('Aloni:2018vki')
    citations.register_inspire('Ovchynnikov:2025gpx')
    cq_eff = cqhat(couplings, ma, **kwargs)
    eps = fpi/np.sqrt(2)/fa
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    return {
        'pi0': eps*(cq_eff[0,0]-cq_eff[1,1])/2,
        'eta': eps*np.sqrt(3)/6*((cq_eff[0,0]+cq_eff[1,1])*(c_eta-np.sqrt(2)*s_eta) - cq_eff[2,2]*(2*c_eta+np.sqrt(2)*s_eta)),
        'etap': eps/2/np.sqrt(3)*((np.sqrt(2)*c_eta + s_eta)*(cq_eff[0,0]+cq_eff[1,1]) + cq_eff[2,2]*(np.sqrt(2)*c_eta-2*s_eta)),
        'K0': eps/np.sqrt(2)*cq_eff[1,2],
        'K0bar': eps/np.sqrt(2)*cq_eff[2,1],
        }

def mass_mixing(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> dict[str, complex]:
    citations.register_inspire('Aloni:2018vki')
    citations.register_inspire('Ovchynnikov:2025gpx')
    eps = fpi/np.sqrt(2)/fa
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    deltaI = (md-mu)/(md+mu)
    cG = couplings['cG']
    return {
        'pi0': -eps*cG*mpi0**2*(kappa[1,1]*(deltaI+1)+kappa[0,0]*(deltaI-1)),
        'eta': -2*eps*cG*mpi0**2/np.sqrt(3)/(mu+md) *((np.sqrt(2)*s_eta-c_eta)*(kappa[0,0]*mu + kappa[1,1]*md)+(2*c_eta+np.sqrt(2)*s_eta)*kappa[2,2]*ms),
        'etap': 2*eps*cG*mpi0**2/np.sqrt(3)/(mu+md) *((s_eta+np.sqrt(2)*c_eta)*(kappa[0,0]*mu + kappa[1,1]*md)+(np.sqrt(2)*c_eta-2*s_eta)*kappa[2,2]*ms),
        'K0': 0,
        'K0bar': 0,
    }

def sm_massmixing():
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    sm_mixing = dict()
    sm_mixing[('pi0', 'eta')] = sm_mixing[('eta', 'pi0')] = - (md-mu)/(md+mu)*mpi0**2*(c_eta-np.sqrt(2)*s_eta)/np.sqrt(3)
    sm_mixing[('pi0', 'etap')] = sm_mixing[('etap', 'pi0')] = - (md-mu)/(md+mu)*mpi0**2*(s_eta+np.sqrt(2)*c_eta)/np.sqrt(3)
    return sm_mixing

def sm_mixingangles():
    res = dict()
    sm_mixing = sm_massmixing()
    #masses = mesonmass_chiPT()
    from ..constants import mpi0, meta, metap, mK0
    masses = {'pi0': mpi0, 'eta': meta, 'etap': metap, 'K0': mK0, 'K0bar': mK0}
    for m1 in ('pi0', 'eta', 'etap'):
        for m2 in ('pi0', 'eta', 'etap'):
            if m1 == m2:
                continue
            res[(m1, m2)] = sm_mixing.get((m1, m2),0)/(masses[m1]**2-masses[m2]**2)
    return res

@lru_cache
def a_U3_repr(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> np.ndarray:
    citations.register_inspire('Aloni:2018vki')
    mesons = ('pi0', 'eta', 'etap', 'K0', 'K0bar')
    sm_mixing = sm_massmixing()
    alp_mixing = dict()
    kmix = kinetic_mixing(ma, couplings, fa, **kwargs)
    mmix = mass_mixing(ma, couplings, fa, **kwargs)
    #mesonmass = mesonmass_chiPT()
    from ..constants import mpi0, meta, metap, mK0
    mesonmass = {'pi0': mpi0, 'eta': meta, 'etap': metap, 'K0': mK0, 'K0bar': mK0}
    for m1 in mesons:
        h = mmix[m1] - ma**2 * kmix[m1]
        for m2 in mesons:
            h += sm_mixing.get((m1, m2), 0) * (mmix[m2] - ma**2 * kmix[m2])/(ma**2-mesonmass[m2]**2)
        h /= (ma**2-mesonmass[m1]**2)
        alp_mixing[m1] = h
    return (u3reprs.pi0 * alp_mixing['pi0'] + u3reprs.eta * alp_mixing['eta'] + u3reprs.etap * alp_mixing['etap'] + u3reprs.K0 * alp_mixing['K0'] + u3reprs.K0bar * alp_mixing['K0bar'])

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
        if ma < 1.45: fun = 1
        elif ma >= 1.45 and ma <= 2: 
            fun = self.ffunction_interp(ma)
        else: fun = (1.4/ma)**4
        return fun
    
ffunction = _ffunction()
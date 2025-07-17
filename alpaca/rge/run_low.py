import numpy as np
import particle.literals
from . import ALPcouplings, runSM
from typing import Callable
from scipy.integrate import solve_ivp
from ..biblio.biblio import citations

def cGtilde(couplings: ALPcouplings) -> complex:
    citations.register_particle()
    cG = couplings['cG']
    if couplings.scale > particle.literals.u.mass/1000:
        cG += 0.5*(couplings['ku'][0,0]-couplings['kU'][0,0])
    if couplings.scale > particle.literals.d.mass/1000:
        cG += 0.5*(couplings['kd'][0,0]-couplings['kD'][0,0])
    if couplings.scale > particle.literals.c.mass/1000:
        cG += 0.5*(couplings['ku'][1,1]-couplings['kU'][1,1])
    if couplings.scale > particle.literals.s.mass/1000:
        cG += 0.5*(couplings['kd'][1,1]-couplings['kD'][1,1])
    if couplings.scale > particle.literals.b.mass/1000:
        cG += 0.5*(couplings['kd'][2,2]-couplings['kD'][2,2])
    return cG

def cgammatilde(couplings: ALPcouplings) -> complex:
    citations.register_particle()
    cG = couplings['cG']
    if couplings.scale > particle.literals.u.mass/1000:
        cG += 3*(2/3)**2*(couplings['ku'][0,0]-couplings['kU'][0,0])
    if couplings.scale > particle.literals.d.mass/1000:
        cG += 3*(-1/3)**2*(couplings['kd'][0,0]-couplings['kD'][0,0])
    if couplings.scale > particle.literals.c.mass/1000:
        cG += 3*(2/3)**2*(couplings['ku'][1,1]-couplings['kU'][1,1])
    if couplings.scale > particle.literals.s.mass/1000:
        cG += 3*(-1/3)**2*(couplings['kd'][1,1]-couplings['kD'][1,1])
    if couplings.scale > particle.literals.b.mass/1000:
        cG += 3*(-1/3)**2*(couplings['kd'][2,2]-couplings['kD'][2,2])
    if couplings.scale > particle.literals.e_minus.mass/1000:
        cG += (couplings['ke'][0,0]-couplings['kE'][0,0])
    if couplings.scale > particle.literals.mu_minus.mass/1000:
        cG += (couplings['ke'][1,1]-couplings['kE'][1,1])
    if couplings.scale > particle.literals.tau_minus.mass/1000:
        cG += (couplings['ke'][2,2]-couplings['kE'][2,2])
    return cG

def beta(couplings: ALPcouplings) -> ALPcouplings:
    parsSM = runSM(couplings.scale)

    beta_d = parsSM['alpha_s']**2/np.pi**2*cGtilde(couplings)+0.75*parsSM['alpha_em']**2/np.pi**2*(-1/3)**2*cgammatilde(couplings)
    beta_u = parsSM['alpha_s']**2/np.pi**2*cGtilde(couplings)+0.75*parsSM['alpha_em']**2/np.pi**2*(2/3)**2*cgammatilde(couplings)
    beta_e = 0.75*parsSM['alpha_em']**2/np.pi**2*cgammatilde(couplings)

    return ALPcouplings({'kd': beta_d*np.eye(3), 'kD': -beta_d*np.eye(3), 'ku': beta_u*np.eye(2), 'kU': -beta_u*np.eye(2), 'ke': beta_e * np.eye(3), 'kE': beta_e*np.eye(3), 'kNu': np.zeros((3,3)), 'cG': 0, 'cgamma': 0}, scale=couplings.scale, basis='RL_below', ew_scale=couplings.ew_scale)


def run_leadinglog(couplings: ALPcouplings, scale_out: float) -> ALPcouplings:
    """Obtain the ALP couplings at a different scale using the leading log approximation
    
    Parameters
    ----------
    couplings : ALPcouplings
        Object containing the ALP couplings at the original scale

    beta : Callable[ALPcouplings, ALPcouplings]
        Function that return the beta function

    scale_out : float
        Final energy scale, in GeV
    """

    result = couplings + beta(couplings) * np.log(scale_out/couplings.scale)
    result.scale = scale_out
    return result

def run_scipy(couplings: ALPcouplings, scale_out: float, scipy_options: dict) -> ALPcouplings:
    """Obtain the ALP couplings at a different scale using scipy's integration
    
    Parameters
    ----------
    couplings : ALPcouplings
        Object containing the ALP couplings at the original scale

    beta : Callable[ALPcouplings, ALPcouplings]
        Function that return the beta function

    scale_out : float
        Final energy scale, in GeV
    """

    citations.register_inspire('Virtanen:2019joe')
    def fun(t0, y):
        return beta(ALPcouplings._fromarray(y, np.exp(t0), 'RL_below', couplings.ew_scale))._toarray()/(16*np.pi**2)
    
    sol = solve_ivp(fun=fun, t_span=(np.log(couplings.scale), np.log(scale_out)), y0=couplings.translate('RL_below')._toarray(), **scipy_options)
    return ALPcouplings._fromarray(sol.y[:,-1], scale_out, 'RL_below', couplings.ew_scale)
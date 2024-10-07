import numpy as np
import particle.literals
from . import ALPcouplings, runSM
from typing import Callable
from scipy.integrate import solve_ivp

def cggtilde(couplings: ALPcouplings) -> complex:
    cgg = couplings['cg']
    if couplings.scale > particle.literals.u.mass/1000:
        cgg += 0.5*(couplings['ku'][0,0]-couplings['kU'][0,0])
    if couplings.scale > particle.literals.d.mass/1000:
        cgg += 0.5*(couplings['kd'][0,0]-couplings['kD'][0,0])
    if couplings.scale > particle.literals.c.mass/1000:
        cgg += 0.5*(couplings['ku'][1,1]-couplings['kU'][1,1])
    if couplings.scale > particle.literals.s.mass/1000:
        cgg += 0.5*(couplings['kd'][1,1]-couplings['kD'][1,1])
    if couplings.scale > particle.literals.b.mass/1000:
        cgg += 0.5*(couplings['kd'][2,2]-couplings['kD'][2,2])
    return cgg

def cgammatilde(couplings: ALPcouplings) -> complex:
    cgg = couplings['cg']
    if couplings.scale > particle.literals.u.mass/1000:
        cgg += 3*(2/3)**2*(couplings['ku'][0,0]-couplings['kU'][0,0])
    if couplings.scale > particle.literals.d.mass/1000:
        cgg += 3*(-1/3)**2*(couplings['kd'][0,0]-couplings['kD'][0,0])
    if couplings.scale > particle.literals.c.mass/1000:
        cgg += 3*(2/3)**2*(couplings['ku'][1,1]-couplings['kU'][1,1])
    if couplings.scale > particle.literals.s.mass/1000:
        cgg += 3*(-1/3)**2*(couplings['kd'][1,1]-couplings['kD'][1,1])
    if couplings.scale > particle.literals.b.mass/1000:
        cgg += 3*(-1/3)**2*(couplings['kd'][2,2]-couplings['kD'][2,2])
    if couplings.scale > particle.literals.e_minus.mass/1000:
        cgg += (couplings['ke'][0,0]-couplings['kE'][0,0])
    if couplings.scale > particle.literals.mu_minus.mass/1000:
        cgg += (couplings['ke'][1,1]-couplings['kE'][1,1])
    if couplings.scale > particle.literals.tau_minus.mass/1000:
        cgg += (couplings['ke'][2,2]-couplings['kE'][2,2])
    return cgg

def beta(couplings: ALPcouplings) -> ALPcouplings:
    parsSM = runSM(couplings.scale)

    beta_d = parsSM['alpha_s']**2/np.pi**2*cggtilde(couplings)+0.75*parsSM['alpha_em']**2/np.pi**2*(-1/3)**2*cgammatilde(couplings)
    beta_u = parsSM['alpha_s']**2/np.pi**2*cggtilde(couplings)+0.75*parsSM['alpha_em']**2/np.pi**2*(2/3)**2*cgammatilde(couplings)
    beta_e = 0.75*parsSM['alpha_em']**2/np.pi**2*cgammatilde(couplings)

    return ALPcouplings({'kd': beta_d*np.eye(3), 'kD': -beta_d*np.eye(3), 'ku': beta_u*np.eye(2), 'kU': -beta_u*np.eye(2), 'ke': beta_e * np.eye(3), 'kE': beta_e*np.eye(3), 'kNu': np.zeros((3,3)), 'cg': 0, 'cgamma': 0}, scale=couplings.scale, basis='kF_below')


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

    result = couplings + beta(couplings) * (np.log(scale_out/couplings.scale)/(16*np.pi**2))
    result.scale = scale_out
    return result


def run_scipy(couplings: ALPcouplings, scale_out: float) -> ALPcouplings:
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

    def fun(t0, y):
        return beta(ALPcouplings._fromarray(y, np.exp(t0), 'kF_below'))._toarray()/(16*np.pi**2)
    
    sol = solve_ivp(fun=fun, t_span=(np.log(couplings.scale), np.log(scale_out)), y0=couplings.translate('kF_below')._toarray())
    return ALPcouplings._fromarray(sol.y[:,-1], scale_out, 'kF_below')
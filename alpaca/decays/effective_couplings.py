from ..rge import ALPcouplings, bases_above, bases_below
from ..constants import mW, s2w, me, mmu, mtau, mc, mb, mu, md, ms, metap, fpi, mt, mb, mpi0, mK, mZ
from ..common import alpha_em, alpha_s, B1, B2
from .alp_decays.chiral import a_U3_repr, ffunction, kappa
from .alp_decays import u3reprs
from ..citations import citations
import numpy as np
from scipy.integrate import quad_vec, quad
import pandas as pd
from scipy.interpolate import interp1d
import os
from functools import cache

def cgamma_chiral(couplings: ALPcouplings) -> float:
    citations.register_inspire('Bauer:2017ris')
    citations.register_inspire('Aloni:2018vki')
    if couplings.scale > metap:
        return 0
    charges = np.diag([2/3, -1/3, -1/3])
    return -2*couplings['cg']*3*np.trace(kappa @ charges @ charges)

def cgamma_VMD(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    citations.register_inspire('Aloni:2018vki')
    citations.register_inspire('Fujiwara:1984mp')
    if ma > 3.0:
        return 0
    a = a_U3_repr(ma, couplings, fa, **kwargs)
    return ffunction(ma)*(3*np.trace(a @ u3reprs.rho0 @ u3reprs.rho0) + 1/3*np.trace(a @ u3reprs.omega @ u3reprs.omega) + 2/3*np.trace(a @ u3reprs.phi @ u3reprs.phi) + 2*np.trace(a @ u3reprs.rho0 @ u3reprs.omega))*fa/fpi

def cgamma_twoloops(couplings: ALPcouplings, fa: float) -> float:
    citations.register_inspire('Bauer:2017ris')
    if couplings['cg'] == 0.0:
        return 0
    ma = couplings.scale
    if ma < metap:
        return 0
    Lambda = np.abs(couplings['cg'])*32*np.pi**2*fa
    charges = [2/3, -1/3, 2/3, -1/3, 2/3, -1/3]
    masses = [mu, md, mc, ms, mt, mb]
    masses_log = [mpi0, mpi0, mc, mK, mt, mb]
    return -3/2*alpha_s(ma)**2/np.pi**2*couplings['cg']*sum(charges[i]**2*B1(4*masses[i]**2/ma**2)*np.log(Lambda**2/masses_log[i]**2) for i in range(6))

def cgamma(couplings: ALPcouplings, fa: float, **kwargs) -> complex:
    cgamma_eff = 0
    ma = couplings.scale
    if couplings.basis in bases_above:
        cc = couplings.translate('massbasis_above')
        cgamma_eff += 2*alpha_em(couplings.scale)/np.pi*cc['cW']/s2w*B2(4*mW**2/ma**2)
        cuA = cc['ku'] - cc['kU']
        cdA = cc['kd'] - cc['kD']
        ceA = cc['ke'] - cc['kE']
        masses = [me, mmu, mtau, mc, mb, mt]
        charges = [-1, -1, -1, 2/3, -1/3, 2/3]
        Nc = [1, 1, 1, 3, 3, 3]
        coups = [ceA[0,0], ceA[1,1], ceA[2,2], cuA[1,1], cdA[2,2], cuA[2,2]]
        cgamma_eff += sum(Nc[i]*charges[i]**2*coups[i]*B1(4*masses[i]**2/ma**2) for i in range(6))
    else:
        cc = couplings.translate('VA_below')
        cuA = cc['cuA']
        cdA = cc['cdA']
        ceA = cc['ceA']
        masses = [me, mmu, mtau, mc, mb]
        charges = [-1, -1, -1, 2/3, -1/3]
        Nc = [1, 1, 1, 3, 3]
        coups = [ceA[0,0], ceA[1,1], ceA[2,2], cuA[1,1], cdA[2,2]]
        cgamma_eff += sum(Nc[i]*charges[i]**2*coups[i]*B1(4*masses[i]**2/ma**2) for i in range(5))
    cgamma_eff += cc['cgamma'] 
    
    if ma > 2.5:
        masses = [mu, md, ms]
        charges = [2/3, -1/3, -1/3]
        coups = [cuA[0,0], cdA[0,0], cdA[1,1]]
        cgamma_eff += sum(charges[i]**2*coups[i]*B1(4*masses[i]**2/ma**2) for i in range(3))*3 + cgamma_twoloops(couplings, fa)
    elif ma > 1.5:
        masses = [mu, md, ms]
        charges = [2/3, -1/3, -1/3]
        coups = [cuA[0,0], cdA[0,0], cdA[1,1]]
        pQCD = sum(charges[i]**2*coups[i]*B1(4*masses[i]**2/ma**2) for i in range(3))*3 + cgamma_twoloops(couplings, fa)
        vmd = cgamma_VMD(ma, couplings, fa, **kwargs)
        interp = -ma + 2.5
        cgamma_eff += interp*vmd - (1-interp)*pQCD
    elif ma > metap:
        cgamma_eff += cgamma_VMD(ma, couplings, fa, **kwargs)
    else:
        cgamma_eff += cgamma_chiral(couplings) + cgamma_VMD(ma, couplings, fa, **kwargs)
    return cgamma_eff

def cg(couplings: ALPcouplings) -> complex:
    citations.register_inspire('Bauer:2017ris')

    ma = couplings.scale
    if couplings.basis in bases_above:
        cc = couplings.translate('massbasis_above')
        cuA = cc['ku'] - cc['kU']
        cdA = cc['kd'] - cc['kD']
    else:
        cc = couplings.translate('VA_below')
        cuA = cc['cuA']
        cdA = cc['cdA']
    mq = [mu, md, ms, mc, mb]
    coupl = [cuA[0,0], cdA[0,0], cdA[1,1], cuA[1,1], cdA[2,2]]
    return cc['cg'] + 0.5 * sum(coupl[i]*B1(4*mq[i]**2/ma**2) for i in range(5))

@cache
def gloop(tau: float) -> complex:
    if tau > 1000:
        return 7/3
    if tau < 0.001:
        return -(np.log(4/tau)-1j*np.pi)**2/6+2/3
    tau0 = (1-1e-10j)*tau
    integrand = lambda x: (1-4*tau0*(1-x)**2-2*x+4*x**2)/np.sqrt(tau0*(1-x)**2-x**2)*np.arctan(x/np.sqrt(tau0*(1-x)**2-x**2))
    integrand_re = lambda x: np.real(integrand(x))
    integrand_im = lambda x: np.imag(integrand(x))
    g_re = quad(integrand_re, 0, 1)[0]
    g_im = quad(integrand_im, 0, 1)[0]
    return 5+4/3*(g_re + 1j*g_im)

def clepton(couplings: ALPcouplings, **kwargs) -> np.matrix:
    citations.register_inspire('Bauer:2017ris')

    deltaI = kwargs.get('deltaI', -11/3)
    ma = couplings.scale
    if couplings.basis in bases_above:
        cc = couplings.translate('massbasis_above')
        ceA = cc['ke'] - cc['kE']
        cWW = cc['cW']
        cgammaZ = cc['cgammaZ']
        cZZ = cc['cZ']
        cgamma = cc['cgamma']
    else:
        cc = couplings.translate('VA_below')
        ceA = cc['ceA']
        cWW = 0
        cgammaZ = 0
        cZZ = 0
        cgamma = cc['cgamma']
    aem = alpha_em(ma)
    mlep = np.array([me, mmu, mtau], dtype=complex)
    ceff = np.array([0,0,0], dtype=complex)
    ql = -1
    t3l = -0.5
    if cgamma != 0:
        ceff -= 12*aem**2*ql**2*cgamma*(np.log(ma**2/mlep**2) + deltaI*np.array([1,1,1], dtype=complex) + np.array([gloop(4*ml**2/ma**2) for ml in mlep]))
    if cWW != 0:
        ceff -= 3*aem**2/s2w**2*cWW*np.array([1,1,1], dtype=complex)*(np.log(ma**2/mW**2)+deltaI+0.5)
    if cgammaZ != 0:
        ceff -= 12*aem**2/s2w/(1-s2w)*cgammaZ*(t3l-2*ql*s2w)*np.array([1,1,1], dtype=complex)*(np.log(ma**2/mZ**2)+deltaI+1.5)
    if cZZ != 0:
        ceff -= 12*aem**2/s2w**2/(1-s2w)**2*cZZ*(ql**2*s2w**2-t3l*ql*s2w+1/8)*np.array([1,1,1], dtype=complex)*(np.log(ma**2/mZ**2)+deltaI+0.5)
    return np.diag(ceff) + ceA
from ...chiPT.chiral import a_U3_repr, kappa, ffunction
from ...chiPT import u3reprs
from ...rge import ALPcouplings, bases_above
from ...constants import mpi0, metap, mK, mu, md, ms, mc, mb, mt, mW, s2w, me, mmu, mtau, fpi
from ...common import alpha_s, alpha_em, B1, B2
from ...biblio.biblio import citations
import numpy as np


def cgamma_chiral(ma: float, couplings: ALPcouplings) -> float:
    citations.register_inspire('Bauer:2017ris')
    citations.register_inspire('Aloni:2018vki')
    if ma > metap:
        return 0
    charges = np.diag([2/3, -1/3, -1/3])
    return -2*couplings['cG']*3*np.trace(kappa @ charges @ charges)

def cgamma_VMD(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    citations.register_inspire('Aloni:2018vki')
    citations.register_inspire('Fujiwara:1984mp')
    if ma > 3.0:
        return 0
    a = a_U3_repr(ma, couplings, fa, **kwargs)
    return ffunction(ma)*(3*np.trace(a @ u3reprs.rho0 @ u3reprs.rho0) + 1/3*np.trace(a @ u3reprs.omega @ u3reprs.omega) + 2/3*np.trace(a @ u3reprs.phi @ u3reprs.phi) + 2*np.trace(a @ u3reprs.rho0 @ u3reprs.omega))*fa/fpi

def cgamma_twoloops(ma: float, couplings: ALPcouplings, fa: float) -> float:
    citations.register_inspire('Bauer:2017ris')
    if couplings['cG'] == 0.0:
        return 0
    if ma < metap:
        return 0
    Lambda = np.abs(couplings['cG'])*32*np.pi**2*fa
    charges = [2/3, -1/3, 2/3, -1/3, 2/3, -1/3]
    masses = [mu, md, mc, ms, mt, mb]
    masses_log = [mpi0, mpi0, mc, mK, mt, mb]
    return -3/2*alpha_s(ma)**2/np.pi**2*couplings['cG']*sum(charges[i]**2*B1(4*masses[i]**2/ma**2)*np.log(Lambda**2/masses_log[i]**2) for i in range(6))

def decay_width_2gamma(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    cgamma_eff = 0
    if ma > couplings.ew_scale:
        cc = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        cgamma_eff += 2*alpha_em/np.pi*cc['cW']/s2w*B2(4*mW**2/ma**2)
        cuA = cc['cuR'] - cc['cuL']
        cdA = cc['cdR'] - cc['cdL']
        ceA = cc['ceR'] - cc['ceL']
        masses = [me, mmu, mtau, mc, mb, mt]
        charges = [-1, -1, -1, 2/3, -1/3, 2/3]
        Nc = [1, 1, 1, 3, 3, 3]
        coups = [ceA[0,0], ceA[1,1], ceA[2,2], cuA[1,1], cdA[2,2], cuA[2,2]]
        cgamma_eff += sum(Nc[i]*charges[i]**2*coups[i]*B1(4*masses[i]**2/ma**2) for i in range(6))
    else:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
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
        cgamma_eff += sum(charges[i]**2*coups[i]*B1(4*masses[i]**2/ma**2) for i in range(3))*3 + cgamma_twoloops(ma, couplings, fa)
    elif ma > 1.5:
        masses = [mu, md, ms]
        charges = [2/3, -1/3, -1/3]
        coups = [cuA[0,0], cdA[0,0], cdA[1,1]]
        pQCD = sum(charges[i]**2*coups[i]*B1(4*masses[i]**2/ma**2) for i in range(3))*3 + cgamma_twoloops(ma, couplings, fa)
        vmd = cgamma_VMD(ma, couplings, fa, **kwargs)
        interp = -ma + 2.5
        cgamma_eff += interp*vmd - (1-interp)*pQCD
    elif ma > metap:
        cgamma_eff += cgamma_VMD(ma, couplings, fa, **kwargs)
    else:
        cgamma_eff += cgamma_chiral(ma, couplings) + cgamma_VMD(ma, couplings, fa, **kwargs)
    return alpha_em(ma)**2*ma**3*np.abs(cgamma_eff)**2/((4*np.pi)**3*fa**2)

def decay_width_2gluons(ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    citations.register_inspire('Bauer:2017ris')
    if ma < 0.6:
        return 0.0

    match_scale = couplings.ew_scale
    if ma > match_scale:
        cc = couplings.match_run(ma, 'massbasis_ew', **kwargs)
        cuA = cc['cuR'] - cc['cuL']
        cdA = cc['cdR'] - cc['cdL']
        mq = [mu, md, ms, mc, mb, mt]
        coupl = [cuA[0,0], cdA[0,0], cdA[1,1], cuA[1,1], cdA[2,2], cuA[2,2]]
        cG_eff = cc['cG'] + 0.5 * sum(coupl[i]*B1(4*mq[i]**2/ma**2) for i in range(6))
    else:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
        cuA = cc['cuA']
        cdA = cc['cdA']
        mq = [mu, md, ms, mc, mb]
        coupl = [cuA[0,0], cdA[0,0], cdA[1,1], cuA[1,1], cdA[2,2]]
        cG_eff = cc['cG'] + 0.5 * sum(coupl[i]*B1(4*mq[i]**2/ma**2) for i in range(5))
    return alpha_s(ma)**2*ma**3/((4*np.pi)**3*fa**2)*np.abs(cG_eff)**2*(1+alpha_s(ma)/np.pi*83/4)#(1+alpha_s(ma)/48/np.pi*(291-sum(14 for i in range(5) if ma > mq[i])))
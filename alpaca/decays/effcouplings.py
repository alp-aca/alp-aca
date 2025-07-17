from ..rge import ALPcouplings
from ..common import B0disc_equalmass, ckm_xi
from ..constants import GF, mu, md, ms, mc, mb, me, mmu, mtau, s2w, mW, mZ
import numpy as np
from ..common import g_photonloop, alpha_em, alpha_s, B3
from ..biblio.biblio import citations

def effcoupling_ff(ma, couplings: ALPcouplings, fermion, **kwargs):
    mass = {'e': me, 'mu': mmu, 'tau': mtau, 'c': mc, 'b': mb}[fermion]
    ftype = {'e': 'e', 'mu': 'e', 'tau': 'e', 'c': 'u', 'b': 'd'}[fermion]
    Nc = {'e': 1, 'mu': 1, 'tau': 1, 'c': 3, 'b': 3}[fermion]
    gen = {'e': 0, 'mu': 1, 'tau': 2, 'c': 1, 'b': 2}[fermion]
    qf = {'e': -1, 'mu': -1, 'tau': -1, 'c': 2/3, 'b': -1/3}[fermion]
    t3f = {'e': -0.5, 'mu': -0.5, 'tau': -0.5, 'c': 0.5, 'b': -0.5}[fermion]
    delta1 = -11/3
    aem = alpha_em(mass**2)/4/np.pi
    if Nc == 3:
        a_s = alpha_s(mass**2)/4/np.pi
    else:
        a_s = 0
    if ma < couplings.ew_scale:
        cc = couplings.match_run(ma, 'VA_below', **kwargs)
        cgamma = cc['cgamma']
        cG = cc['cG']
        cf = cc[f'c{ftype}A'][gen, gen]
        if cgamma != 0 or (cG!=0 and Nc == 3):
            g = g_photonloop(4*mass**2/ma**2)
        else:
            g = 0
        ceff = cf
        ceff -= 12 * qf**2 * aem**2 * cgamma * (np.log(ma**2/mass**2) + delta1+g)
        if Nc == 3:
            ceff -= 12 * (4/3) * a_s**2 * cG * (np.log(ma**2/mass**2) + delta1+g)
        return ceff
    else:
        cc = couplings.translate('massbasis_ew')
        cgamma = cc['cgamma']
        cgammaZ = cc['cgammaZ']
        cZ = cc['cZ']
        cW = cc['cW']
        cG = cc['cG']
        cf = cc[f'k{ftype}'][gen, gen]-cc[f'k{ftype.upper()}'][gen, gen]
        if cgamma != 0 or (cG!=0 and Nc == 3):
            g = g_photonloop(4*mass**2/ma**2)
        else:
            g = 0
        ceff = cf
        ceff -= 12 * qf**2 * aem**2 * cgamma * (np.log(ma**2/mass**2) + delta1+g)
        if Nc == 3:
            ceff -= 12 * (4/3) * a_s**2 * cG * (np.log(ma**2/mass**2) + delta1+g)
        c2w = 1-s2w
        ceff -= 3* aem**2/s2w**2 * cW * (np.log(ma**2/mW**2) + delta1 + 1/2)
        ceff -= 12*aem**2/s2w/c2w * cgammaZ * qf *(t3f - 2*qf*s2w)* (np.log(ma**2/mZ**2) + delta1 + 3/2)
        ceff -= 12*aem**2/s2w**2/c2w**2 * cZ * (qf**2*s2w**2-t3f*qf*s2w+1/8) * (np.log(ma**2/mZ**2) + delta1 + 1/2)
        return ceff

def effcouplings_cq1q2_W(couplings: ALPcouplings, pa2: float, q1: str, q2: str) -> complex:
    if couplings.scale > couplings.ew_scale:
        raise NotImplementedError(f"The effective couplings c_{q1}{q2} are implemented only below the EW scale.")
    couplings = couplings.translate('RL_below')
    mq = {'u': mu, 'd': md, 's': ms, 'c': mc, 'b': mb}
    ceff = 0
    if q1 == q2:
        return ceff
    if q1 in ['u', 'c'] and q2 in ['u', 'c']:
        gen = {'u': 0, 'c': 1}
        ceff = couplings['cuL'][gen[q1], gen[q2]]
        for iq, qloop in enumerate(['d', 's', 'b']):
            cqloop = couplings['cdL'][iq, iq] - couplings['cdR'][iq, iq]
            ceff += GF/np.sqrt(2)/np.pi**2*ckm_xi(qloop, q1+q2)*cqloop * mq[qloop]**2 * (1 + B0disc_equalmass(pa2, mq[qloop]) + np.log(couplings.scale**2/mq[qloop]**2))
    elif q1 in ['d', 's', 'b'] and q2 in ['d', 's', 'b']:
        gen = {'d': 0, 's': 1, 'b': 2}
        ceff = couplings['cdL'][gen[q1], gen[q2]]
        for iq, qloop in enumerate(['u', 'c']):
            cqloop = couplings['cuL'][iq, iq] - couplings['cuR'][iq, iq]
            ceff += GF/np.sqrt(2)/np.pi**2*ckm_xi(qloop, q1+q2) * cqloop * mq[qloop]**2 * (1 + B0disc_equalmass(pa2, mq[qloop]) + np.log(couplings.scale**2/mq[qloop]**2))
    return ceff

def offshellphoton(couplings: ALPcouplings, ma: float, s: float) -> complex:
    """Effective coupling of the ALP to one on-shell and one off-shell photon."""
    if couplings.scale > couplings.ew_scale:
        raise NotImplementedError("The effective coupling of the ALP to one on-shell and one off-shell photon is implemented only below the EW scale.")
    citations.register_inspire('Alda:2024cxn')
    couplings = couplings.translate('VA_below')
    ceff = couplings['cgamma']
    for i, mlep in enumerate([me, mmu, mtau]):
        ceff += couplings['ceA'][i,i] * B3(4*mlep**2/ma**2, 4*mlep**2/s)
    for i, muq in enumerate([mu, mc]):
        ceff += 3 * (2/3)**2 * couplings['cuA'][i,i] * B3(4*muq**2/ma**2, 4*muq**2/s)
    for i, mdq in enumerate([md, ms, mb]):
        ceff += 3 * (-1/3)**2 * couplings['cdA'][i,i] * B3(4*mdq**2/ma**2, 4*mdq**2/s)
    return ceff
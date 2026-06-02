from ..rge import ALPcouplings
from ..rge.runSM import runSM
from ..common import B0disc_equalmass, ckm_xi
from ..constants import GF, mu, md, ms, mc, mb, mt, me, mmu, mtau, s2w, mW, mZ, mH
import numpy as np
from ..common import g_photonloop, alpha_em, alpha_s, B3, B0disc_lim, B0disc_equalmass, floop
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
        cc = couplings.translate('derivative_above')
        smpars = runSM(ma)
        s2w = smpars['s2w']
        c2w = 1-s2w
        cgamma = cc['cB'] + cc['cW']
        cgammaZ = c2w * cc['cW'] - s2w * cc['cB']
        cZ = c2w**2 * cc['cW'] + s2w**2 * cc['cB']
        cW = cc['cW']
        cG = cc['cG']
        doublet = {'e': 'l', 'd': 'q', 'u': 'q'}[ftype]
        cf = cc[f'c{ftype}R'][gen, gen]-cc[f'c{doublet}L'][gen, gen]
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

def effcoupling_gammaZ(couplings: ALPcouplings, ma: float) -> complex:
    """Effective coupling of the ALP to one on-shell photon and one on-shell Z boson."""
    if couplings.scale < couplings.ew_scale:
        raise NotImplementedError("The effective coupling of the ALP to one on-shell photon and one on-shell Z boson is implemented only above the EW scale.")
    citations.register_inspire('Bonilla:2021ufe')
    couplings2 = couplings.copy()
    scale = max(ma, mZ)
    couplings2.ew_scale = scale
    couplings2 = couplings2.match_run(scale, 'massbasis_ew')
    ceff = couplings2['cgammaZ']
    c2w = 1 - s2w

    fermions = ['u', 'd', 's', 'c', 'b', 't', 'e', 'mu', 'tau']
    nc = {q: 3 for q in ['u', 'd', 's', 'c', 'b', 't']} | {f: 1 for f in ['e', 'mu', 'tau']}
    q = {q: 2/3 for q in ['u', 'c', 't']} | {q: -1/3 for q in ['d', 's', 'b']} | {f: -1 for f in ['e', 'mu', 'tau']}
    t3 = {q: 1/2 for q in ['u', 'c', 't']} | {q: -1/2 for q in ['d', 's', 'b']} | {f: -1/2 for f in ['e', 'mu', 'tau']}
    m = {**{f: eval(f'm{f}') for f in ['u', 'd', 's', 'c', 'b', 't']}, **{f: eval(f'm{f}') for f in ['e', 'mu', 'tau']}}
    gen = {f: 0 for f in ['u', 'd', 'e']} | {f: 1 for f in ['s', 'c', 'mu']} | {f: 2 for f in ['b', 't', 'tau']}
    ptype = {f: 'u' for f in ['u', 'c', 't']} | {f: 'd' for f in ['d', 's', 'b']} | {f: 'e' for f in ['e', 'mu', 'tau']}

    a_Zgamma_gamma = 1 - 2*sum(q[f]**2 * nc[f] * np.log(ma**2/m[f]**2) for f in fermions) + 21/2 * np.log(ma**2/mW**2)

    a_Zgamma_Zf = 0
    for f in fermions:
        a = (t3[f]**2/2 - t3[f]*q[f]*s2w + q[f]**2*s2w**2) * (np.log(ma**2/m[f]**2) +2/3 + (mZ**2 - 2*m[f]**2)/(mZ**2 - 4*m[f]**2) * B0disc_equalmass(mZ**2, m[f]))
        a += m[f]**2/mZ**2 * (t3[f]**2/2 + 2*t3[f]*q[f]*s2w -2*q[f]**2*s2w**2) * (1-2*m[f]**2/(mZ**2-4*m[f]**2)*B0disc_equalmass(mZ**2, m[f]))
        a -= c2w*q[f]*(t3[f]-2*q[f]*s2w) * (np.log(ma**2/m[f]**2) + (12*m[f]**2+5*mZ**2)/(3*mZ**2) + (mZ**2 + 2*m[f]**2)/mZ**2 * B0disc_equalmass(mZ**2, m[f]))
        a_Zgamma_Zf -= 2*nc[f]*a

    a_ZZ_h = (mZ**4-2*mZ**2*mH**2+mH**4)/mZ**4
    a_ZZ_h += 0.25*(12*mZ**6-18*mZ**4*mH**2+9*mZ**2*mH**4-2*mH**6)/mZ**6 * np.log(mH**2/mZ**2)
    a_ZZ_h -= (36*mZ**6-32*mZ**4*mH**2+13*mZ**2*mH**4-2*mH**6)/(2*mZ**4*(mH**2-4*mZ**2)) * B0disc_lim(mZ, mH)

    a_Zgamma_gauge = 0.5*(42*mW**4+mZ**4)/mZ**4 * np.log(ma**2/mW**2)
    a_Zgamma_gauge += 0.25*mW**4/mZ**4 * np.log(mW**2/mZ**2)
    a_Zgamma_gauge += (180*mW**6+153*mW**4*mZ**2-12*mW**2*mZ**4-5*mZ**6)/mZ**6/3
    a_Zgamma_gauge += 0.25*(120*mW**6+108*mW**4*mZ**2+2*mW**2*mZ**4+mZ**6)/mZ**6 * B0disc_equalmass(mZ**2, mW)
    a_Zgamma_gauge *= -0.5

    ceff *= 1 + alpha_em(ma)/(12*np.pi) *(a_Zgamma_gamma + (a_Zgamma_Zf + a_ZZ_h + a_Zgamma_gauge)/s2w/c2w)

    a_WW = (42*mW**2+mZ**2)/(12*mW**2)*np.log(ma**2/mW**2)
    a_WW += (36*mW**4+93*mW**2*mZ**2+2*mZ**4)/(9*mW**2*mZ**2)
    a_WW += (24*mW**4+38*mW**2*mZ**2+mZ**4)/(12*mW**2*mZ**2) * B0disc_equalmass(mZ**2, mW)
    a_WW -= 4*(4*mW**2-ma**2)/(ma**2-mZ**2)*(floop(4*mW**2/ma**2)**2 - floop(4*mW**2/mZ**2)**2)
    for f in fermions:
        a_WW -= nc[f]*q[f]*(t3[f]-2*q[f]*s2w)/(3*c2w) * (np.log(ma**2/m[f]**2) + (12*m[f]**2+5*mZ**2)/(3*mZ**2) + (mZ**2 + 2*m[f]**2)/mZ**2 * B0disc_equalmass(mZ**2, m[f]))

    ceff += 0.5 * c2w/s2w * couplings2['cW'] * a_WW

    for f in fermions:
        ceff += (couplings2[f'c{ptype[f]}R'][gen[f],gen[f]] - couplings2[f'c{ptype[f]}L'][gen[f],gen[f]]) * q[f] * nc[f] *(2*q[f]*s2w + 4*(t3[f]-2*q[f]*s2w)*m[f]/(ma**2-mZ**2)*(floop(4*m[f]**2/ma**2)**2 - floop(4*m[f]**2/mZ**2)**2))

    return ceff
import numpy as np
from . import ALPcouplings, runSM
import particle.literals
from ..biblio.biblio import citations

def gauge_tilde(couplings):
        parsSM = runSM(couplings.scale)
        s2w = parsSM['s2w']
        c2w = 1-s2w
        dcW = - 0.5*np.trace(3*couplings['cuL']+couplings['ceL'])
        dcB = np.trace(4/3*couplings['cuR']+couplings['cdR']/3-couplings['cuL']/6+couplings['ceR']-couplings['ceL']/2)
        cW = couplings['cW'] + dcW
        cZ = couplings['cZ'] + c2w**2 * dcW + s2w**2 * dcB
        cgammaZ = couplings['cgammaZ'] + c2w *dcW - s2w * dcB
        cgamma = couplings['cgamma'] + dcW + dcB
        return {'cWtilde': cW, 'cZtilde': cZ, 'cgammatilde': cgamma, 'cgammaZtilde': cgammaZ}

def match_FCNC_d(couplings: ALPcouplings, two_loops = False, loopquark=2) -> np.matrix:
    citations.register_particle()
    mquark = [particle.literals.u.mass,particle.literals.c.mass,particle.literals.t.mass][loopquark]/1000
    mW = particle.literals.W_minus.mass / 1000

    parsSM = runSM(couplings.scale)
    s2w = parsSM['s2w']
    yt = np.real(couplings.yu[loopquark,loopquark])
    alpha_em = parsSM['alpha_em']
    Vckm = np.matrix(couplings.get_ckm())
    if two_loops:
        tildes = gauge_tilde(couplings)
        cW = tildes['cWtilde']
    else:
        cW = couplings['cW']

    xt = mquark**2/mW**2
    Vtop = np.einsum('ia,bj->ijab', Vckm.H, Vckm)[:,:,loopquark,loopquark]  # V_{ti}^* V_{tj}
    gx = (1-xt+np.log(xt))/(1-xt)**2
    logm = np.log(couplings.scale**2/mquark**2)
    kFCNC = 0
    kFCNC += (np.einsum('im,nj,mn->ijm', Vckm.H, Vckm, couplings['cuL'])[:,:,2] + np.einsum('im,nj,mn->ijn', Vckm.H, Vckm, couplings['cuL'])[:,:,2]) * (-0.25*logm-0.375+0.75*gx)
    kFCNC += Vtop*couplings['cuL'][2,2]
    kFCNC += Vtop*couplings['cuR'][2,2]*(0.5*logm-0.25-1.5*gx)
    kFCNC -= 1.5*alpha_em/np.pi/s2w * cW * Vtop * (1-xt+xt*np.log(xt))/(1-xt)**2
    return yt**2/(16*np.pi**2) * kFCNC

def match(couplings: ALPcouplings, two_loops = False) -> ALPcouplings:
    citations.register_particle()
    T3f = {'uL': 1/2, 'dL': -1/2, 'nuL': 1/2, 'eL': -1/2, 'uR': 0, 'dR': 0, 'eR': 0}
    Qf = {'uL': 2/3, 'dL': -1/3, 'nuL': 0, 'eL': -1, 'uR': 2/3, 'dR': -1/3, 'eR': -1}
    mtop = particle.literals.t.mass / 1000
    mW = particle.literals.W_minus.mass / 1000
    mZ = particle.literals.Z_0.mass / 1000

    parsSM = runSM(couplings.scale)
    s2w = parsSM['s2w']
    c2w = 1-s2w
    yt = np.real(parsSM['yu'][2,2])
    alpha_em = parsSM['alpha_em']

    delta1 = -11/3

    ctt = couplings['cuR'][2,2] - couplings['cuL'][2,2]
    if two_loops:
        tildes = gauge_tilde(couplings)
        cW = tildes['cWtilde']
        cZ = tildes['cZtilde']
        cgammaZ = tildes['cgammaZtilde']

    else:
        cW = couplings['cW']
        cZ = couplings['cZ']
        cgammaZ = couplings['cgammaZ']

    Delta_kF = {F: yt**2*ctt*(T3f[F]-Qf[F]*s2w)*np.log(couplings.scale**2/mtop**2) + \
        alpha_em**2*(0.5*cW/s2w**2*(np.log(couplings.scale**2/mW**2) + 0.5 + delta1) + 2*cgammaZ/s2w/c2w*Qf[F]*(T3f[F]-Qf[F] * s2w)*(np.log(couplings.scale**2/mZ**2) + 1.5 + delta1) + cZ/s2w**2/c2w**2 *(T3f[F]-Qf[F]*s2w)**2 *(np.log(couplings.scale**2/mZ**2) + 0.5 + delta1) ) for F in ['uL', 'dL', 'nuL', 'eL', 'uR', 'dR', 'eR']}

    values = {f'c{F}': couplings[f'c{F}'] + 3/(8*np.pi**2)*Delta_kF[F]*np.eye(3) for F in ['nuL', 'eL', 'dR', 'eR']}
    values |= {f'c{F}': couplings[f'c{F}'][0:2,0:2] + 3/(8*np.pi**2)*Delta_kF[F]*np.eye(2) for F in ['uL', 'uR']}
    values |= {'cdL': couplings['cdL'] + 3/(8*np.pi**2)*Delta_kF['dL']*np.eye(3) + match_FCNC_d(couplings, two_loops)}
    values |= {'cG': couplings['cG'], 'cgamma': couplings['cgamma']}
    return ALPcouplings(values, scale=couplings.scale, basis='RL_below', ew_scale=couplings.ew_scale)


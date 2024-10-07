import numpy as np
from . import ALPcouplings, runSM
import particle.literals

def gauge_tilde(couplings):
        parsSM = runSM(couplings.scale)
        s2w = parsSM['s2w']
        c2w = 1-s2w
        dcW = - 0.5*np.trace(3*couplings['kU']+couplings['kE'])
        dcB = np.trace(4/3*couplings['ku']+couplings['kd']/3-couplings['kU']/6+couplings['ke']-couplings['kE']/2)
        cW = couplings['cW'] + dcW
        cZ = couplings['cZ'] + c2w**2 * dcW + s2w**2 * dcB
        cgammaZ = couplings['cgammaZ'] + c2w *dcW - s2w * dcB
        cgamma = couplings['cgamma'] + dcW + dcB
        return {'cWtilde': cW, 'cZtilde': cZ, 'cgammatilde': cgamma, 'cgammaZtilde': cgammaZ}

def match_FCNC(couplings: ALPcouplings, two_loops = False) -> np.matrix:
    mtop = particle.literals.t.mass / 1000
    mW = particle.literals.W_minus.mass / 1000

    parsSM = runSM(couplings.scale)
    s2w = parsSM['s2w']
    yt = np.real(parsSM['yu'][2,2])
    alpha_em = parsSM['alpha_em']
    Vckm = parsSM['CKM']

    if two_loops:
        tildes = gauge_tilde(couplings)
        cW = tildes['cWtilde']
    else:
        cW = couplings['cW']

    xt = mtop**2/mW**2
    Vtop = np.einsum('ia,bj->ijab', Vckm.H, Vckm)[:,:,2,2]  # V_{ti}^* V_{tj}
    gx = (1-xt+np.log(xt))/(1-xt)**2
    logm = np.log(couplings.scale**2/mtop**2)
    kFCNC = 0
    kFCNC += (np.einsum('im,nj,mn->ijm', Vckm.H, Vckm, couplings['kU'])[:,:,2] + np.einsum('im,nj,mn->ijn', Vckm.H, Vckm, couplings['kU'])[:,:,2]) * (-0.25*logm-0.375+0.75*gx)
    kFCNC += Vtop*couplings['kU'][2,2]
    kFCNC += Vtop*couplings['ku'][2,2]*(0.5*logm-0.25-1.5*gx)
    kFCNC -= 1.5*alpha_em/np.pi/s2w * cW * Vtop * (1-xt+xt*np.log(xt))/(1-xt)**2
    return yt**2/(16*np.pi**2) * kFCNC

def match(couplings: ALPcouplings, two_loops = False) -> ALPcouplings:
    T3f = {'U': 1/2, 'D': -1/2, 'Nu': 1/2, 'E': -1/2, 'u': 0, 'd': 0, 'e': 0}
    Qf = {'U': 2/3, 'D': -1/3, 'Nu': 0, 'E': -1, 'u': 2/3, 'd': -1/3, 'e': -1}
    mtop = particle.literals.t.mass / 1000
    mW = particle.literals.W_minus.mass / 1000
    mZ = particle.literals.Z_0.mass / 1000

    parsSM = runSM(couplings.scale)
    s2w = parsSM['s2w']
    c2w = 1-s2w
    yt = np.real(parsSM['yu'][2,2])
    alpha_em = parsSM['alpha_em']
    Vckm = parsSM['CKM']

    delta1 = -11/3

    ctt = couplings['ku'][2,2] - couplings['kU'][2,2]
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
        alpha_em**2*(0.5*cW/s2w**2*(np.log(couplings.scale**2/mW**2) + 0.5 + delta1) + 2*cgammaZ/s2w/c2w*Qf[F]*(T3f[F]-Qf[F] * s2w)*(np.log(couplings.scale**2/mZ**2) + 1.5 + delta1) + cZ/s2w**2/c2w**2 *(T3f[F]-Qf[F]*s2w)**2 *(np.log(couplings.scale**2/mZ**2) + 0.5 + delta1) ) for F in ['U', 'D', 'Nu', 'E', 'u', 'd', 'e']}

    values = {f'k{F}': couplings[f'k{F}'] + 3/(8*np.pi**2)*Delta_kF[F]*np.eye(3) for F in ['Nu', 'E', 'd', 'e']}
    values |= {f'k{F}': couplings[f'k{F}'][0:2,0:2] + 3/(8*np.pi**2)*Delta_kF[F]*np.eye(2) for F in ['U', 'u']}
    values |= {'kD': couplings['kD'] + 3/(8*np.pi**2)*Delta_kF['D']*np.eye(3) + match_FCNC(couplings, two_loops)}
    return ALPcouplings(values, scale=couplings.scale, basis='kF_below')


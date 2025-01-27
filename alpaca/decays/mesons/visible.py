from ...constants import GF, mB0, mBs, fB, fBs, C10, me, mmu, mtau, GammaB, GammaBs, DeltaGamma_Bs, mK0, fK0, epsilonKaon, phiepsilonKaon, mKL, mKS, GammaKL, GammaKS, C10sdRe, C10sdIm, C7, lambdaB0, mD0, fD0, GammaD0
from ...common import kallen, ckm_xi, alpha_em
from ...rge.classes import ALPcouplings
from ..alp_decays.branching_ratios import total_decay_width
import numpy as np
from flavio.physics.kdecays.kll import amplitudes_LD
from ...biblio.biblio import citations

mlepton = {'e': me, 'mu': mmu, 'tau': mtau}
genlepton = {'e': 0, 'mu': 1, 'tau': 2}

def amp_Bs_leptons_SM(lepton: str) -> complex:
    return -1*GF*alpha_em(mBs)/np.pi*mlepton[lepton]*mBs*fBs*ckm_xi('t', 'bs')*C10

def amp_Bs_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    if ma > couplings.ew_scale:
        basis = 'massbasis_above'
    else:
        basis = 'kF_below'
    cc = couplings.match_run(ma, basis, **kwargs)
    clep = cc['ke'][genlepton[lepton],genlepton[lepton]] - cc['kE'][genlepton[lepton],genlepton[lepton]]
    cbs = np.array([cc['kD'][2,1] - cc['kd'][2,1], cc['kD'][1,2] - cc['kd'][1,2]])
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return - cbs*clep/fa**2 *fBs*mlepton[lepton]/np.sqrt(2)*mBs**3/(mBs**2-ma**2+1j*ma*Gamma_a)

def BR_Bs_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_SM = amp_Bs_leptons_SM(lepton)
    amp_ALP = amp_Bs_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_sq = 0.5*np.abs(amp_SM + amp_ALP[0])**2 + 0.5*np.abs(np.conj(amp_SM) + amp_ALP[1])**2
    gamma_th = amp_sq*np.sqrt(kallen(mBs**2, mlepton[lepton]**2, mlepton[lepton]**2))/(16*np.pi*mBs**3)
    gamma_exp = gamma_th/(1-DeltaGamma_Bs/2)
    return gamma_exp/GammaBs

def amp_Bd_leptons_SM(lepton: str) -> complex:
    return -1*GF*alpha_em(mB0)/np.pi*mlepton[lepton]*mB0*fB*ckm_xi('t', 'bd')*C10

def amp_Bd_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    if ma > couplings.ew_scale:
        basis = 'massbasis_above'
    else:
        basis = 'kF_below'
    cc = couplings.match_run(ma, basis, **kwargs)
    clep = cc['ke'][genlepton[lepton],genlepton[lepton]] - cc['kE'][genlepton[lepton],genlepton[lepton]]
    cbs = np.array([cc['kD'][2,0] - cc['kd'][2,0], cc['kD'][0,2] - cc['kd'][0,2]])
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return - cbs*clep/fa**2 *fB*mlepton[lepton]/np.sqrt(2)*mB0**3/(mB0**2-ma**2+1j*ma*Gamma_a)

def BR_Bd_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_SM = amp_Bd_leptons_SM(lepton)
    amp_ALP = amp_Bd_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_sq = 0.5*np.abs(amp_SM + amp_ALP[0])**2 + 0.5*np.abs(np.conj(amp_SM) + amp_ALP[1])**2
    gamma_th = amp_sq*np.sqrt(kallen(mB0**2, mlepton[lepton]**2, mlepton[lepton]**2))/(16*np.pi*mB0**3)
    return gamma_th/GammaB #The mixing correction is negligible for Bd

def amp_Bs_photons_SM() -> complex:
    citations.register_inspire('Bosch:2002bv')
    return -np.sqrt(2)/3*GF*alpha_em(mBs)/np.pi*mBs/lambdaB0*fBs*ckm_xi('t', 'bs')*C7

def amp_Bs_photons_ALP(ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    if ma > couplings.ew_scale:
        basis = 'massbasis_above'
    else:
        basis = 'kF_below'
    cc = couplings.match_run(ma, basis, **kwargs)
    cbs = np.array([cc['kD'][2,1] - cc['kd'][2,1], cc['kD'][1,2] - cc['kd'][1,2]])
    cgamma = cc['cgamma']
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return 0.5*alpha_em(mBs)/np.pi * cgamma*cbs/fa**2 *fBs*mBs**3/(mBs**2-ma**2+1j*ma*Gamma_a)

def BR_Bs_photons_ALP(ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_SM = amp_Bs_photons_SM()
    amp_ALP = amp_Bs_photons_ALP(ma, couplings, fa, br_dark, **kwargs)
    amp_sq = 0.5*np.abs(amp_SM + amp_ALP[0])**2 + 0.5*np.abs(np.conj(amp_SM) + amp_ALP[1])**2 + np.abs(amp_SM)**2
    gamma_th = amp_sq*mBs**3/(64*np.pi)
    gamma_exp = gamma_th/(1-DeltaGamma_Bs/2)
    return gamma_exp/GammaBs

def amp_Bd_photons_SM() -> complex:
    citations.register_inspire('Bosch:2002bv')
    return -np.sqrt(2)/3*GF*alpha_em(mB0)/np.pi*mB0/lambdaB0*fB*ckm_xi('t', 'bd')*C7

def amp_Bd_photons_ALP(ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    if ma > couplings.ew_scale:
        basis = 'massbasis_above'
    else:
        basis = 'kF_below'
    cc = couplings.match_run(ma, basis, **kwargs)
    cbd = np.array([cc['kD'][2,0] - cc['kd'][2,0], cc['kD'][0,2] - cc['kd'][0,2]])
    cgamma = cc['cgamma']
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return 0.5*alpha_em(mB0)/np.pi * cgamma*cbd/fa**2 *fB*mB0**3/(mB0**2-ma**2+1j*ma*Gamma_a)

def BR_Bd_photons_ALP(ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_SM = amp_Bd_photons_SM()
    amp_ALP = amp_Bd_photons_ALP(ma, couplings, fa, br_dark, **kwargs)
    amp_sq = 0.5*np.abs(amp_SM + amp_ALP[0])**2 + 0.5*np.abs(np.conj(amp_SM) + amp_ALP[1])**2 + np.abs(amp_SM)**2
    gamma_th = amp_sq*mB0**3/(64*np.pi)
    return gamma_th/GammaB #The mixing correction is negligible for Bd

def amp_D0_photons_ALP(ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    if ma > couplings.ew_scale:
        basis = 'massbasis_above'
    else:
        basis = 'kF_below'
    citations.register_inspire('Burdman:2001tf')
    cc = couplings.match_run(ma, basis, **kwargs)
    ccu = cc['kU'][1,0] - cc['ku'][1,0]
    cgamma = cc['cgamma']
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return 0.5*alpha_em(mD0)/np.pi * cgamma*ccu/fa**2 *fD0*mD0**3/(mD0**2-ma**2+1j*ma*Gamma_a)

def BR_D0_photons_ALP(ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    from ...constants import b_D0gammagamma_VMD, c_D0gammagamma_VMD
    amp_ALP = amp_D0_photons_ALP(ma, couplings, fa, br_dark, **kwargs)
    amp_sq = (np.abs(amp_ALP) + c_D0gammagamma_VMD)**2 + b_D0gammagamma_VMD**2
    gamma_th = amp_sq*mD0**3/(64*np.pi)
    return gamma_th/GammaD0

def amp_K0_leptons_SM(lepton: str) -> complex:
    #a_em = alpha_em(mK0)
    a_em = 1/137
    return -1*GF*a_em/np.pi*mlepton[lepton]*mK0*fK0*ckm_xi('t', 'sd')*(C10sdRe+1j*C10sdIm)

def amp_KL_leptons_LD(lepton: str) -> complex:
    from ...constants import re_ae, re_amu, br_KLgammagamma
    citations.register_inspire('Hoferichter:2023wiy')
    #a_em = alpha_em(mK0)
    a_em = 1/137
    beta_l = np.sqrt(1-4*mlepton[lepton]**2/mK0**2)
    im_al = 0.5*np.pi/beta_l*np.log((1-beta_l)/(1+beta_l))
    if lepton == 'e':
        re_al = re_ae
    elif lepton == 'mu':
        re_al = re_amu
    al = re_al + 1j*im_al
    return a_em * al * mlepton[lepton] * np.sqrt(32/np.pi/mK0 * br_KLgammagamma * GammaKL)

def amp_KS_leptons_LD(lepton: str) -> complex:
    from ...constants import g8, ie_abs, ie_disp, imu_abs, imu_disp, mpi0
    citations.register_inspire('Ecker:1991ru')
    #a_em = alpha_em(mK0)
    a_em = 1/137
    if lepton == 'e':
        il = ie_abs + 1j*ie_disp
    elif lepton == 'mu':
        il = imu_abs + 1j*imu_disp
    G8  = -GF/np.sqrt(2)*g8*ckm_xi('u', 'sd')
    bl = a_em**2/np.pi**2 * il * G8 *fK0*mlepton[lepton]*(1-mpi0**2/mK0**2)
    return bl*np.sqrt(2)*mK0

def amp_K0_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    if ma > couplings.ew_scale:
        basis = 'massbasis_above'
    else:
        basis = 'kF_below'
    cc = couplings.match_run(ma, basis, **kwargs)
    clep = cc['ke'][genlepton[lepton],genlepton[lepton]] - cc['kE'][genlepton[lepton],genlepton[lepton]]
    csd = np.array([cc['kD'][1,0] - cc['kd'][1,0], cc['kD'][0,1] - cc['kd'][0,1]])
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return - csd*clep/fa**2 *fK0*mlepton[lepton]/np.sqrt(2)*mK0**3/(mK0**2-ma**2+1j*ma*Gamma_a)

def amp_KL_leptons_P(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    a_ALP = amp_K0_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    a_SM = amp_K0_leptons_SM(lepton)
    amp_K0 = a_SM + a_ALP[0]
    amp_K0bar = np.conj(a_SM) + a_ALP[1]
    eps = epsilonKaon*(np.cos(phiepsilonKaon)+1j*np.sin(phiepsilonKaon))
    amp = ((1+eps)*amp_K0+(1-eps)*amp_K0bar)/np.sqrt(2*(1+np.abs(eps)**2))
    return amp + amp_KL_leptons_LD(lepton)

def amp_KS_leptons_P(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    a_ALP = amp_K0_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    a_SM = amp_K0_leptons_SM(lepton)
    amp_K0 = a_SM + a_ALP[0]
    amp_K0bar = np.conj(a_SM) + a_ALP[1]
    eps = epsilonKaon*(np.cos(phiepsilonKaon)+1j*np.sin(phiepsilonKaon))
    amp = ((1+eps)*amp_K0-(1-eps)*amp_K0bar)/np.sqrt(2*(1+np.abs(eps)**2))
    return amp

def BR_KL_leptons(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_P = amp_KL_leptons_P(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_sq = np.abs(amp_P)**2
    gamma = amp_sq*np.sqrt(kallen(mKL**2, mlepton[lepton]**2, mlepton[lepton]**2))/(16*np.pi*mKL**3)
    return gamma/GammaKL

def BR_KS_leptons(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_P = amp_KS_leptons_P(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_S = amp_KS_leptons_LD(lepton)
    amp_sq = np.abs(amp_P)**2 + np.abs(amp_S)**2 * (1-4*mlepton[lepton]**2/mKS**2)
    gamma = amp_sq*np.sqrt(kallen(mKS**2, mlepton[lepton]**2, mlepton[lepton]**2))/(16*np.pi*mKS**3)
    return gamma/GammaKS
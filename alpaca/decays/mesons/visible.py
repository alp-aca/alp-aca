from ...constants import GF, mB0, mBs, fB, fBs, C10, me, mmu, mtau, GammaB, GammaBs, DeltaGamma_Bs, mK0, fK0, epsilonKaon, phiepsilonKaon, mKL, mKS, GammaKL, GammaKS, C10sdRe, C10sdIm, pars
from ...common import kallen, ckm_xi, alpha_em
from ...rge.classes import ALPcouplings
from ..alp_decays.branching_ratios import total_decay_width
import numpy as np
from flavio.physics.kdecays.kll import amplitudes_LD

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
    cbs = cc['kD'][2,1] - cc['kd'][2,1]
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return - cbs*clep/fa**2 *fBs*mlepton[lepton]/np.sqrt(2)*mBs**3/(mBs**2-ma**2+1j*ma*Gamma_a)

def BR_Bs_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_SM = amp_Bs_leptons_SM(lepton)
    amp_ALP = amp_Bs_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_sq = np.abs(amp_SM + amp_ALP)**2
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
    cbs = cc['kD'][2,0] - cc['kd'][2,0]
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return - cbs*clep/fa**2 *fB*mlepton[lepton]/np.sqrt(2)*mB0**3/(mB0**2-ma**2+1j*ma*Gamma_a)

def BR_Bd_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_SM = amp_Bd_leptons_SM(lepton)
    amp_ALP = amp_Bd_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_sq = np.abs(amp_SM + amp_ALP)**2
    gamma_th = amp_sq*np.sqrt(kallen(mB0**2, mlepton[lepton]**2, mlepton[lepton]**2))/(16*np.pi*mB0**3)
    return gamma_th/GammaB #The mixing correction is negligible for Bd

def amp_K0_leptons_SM(lepton: str) -> complex:
    a_em = alpha_em(mK0)
    #a_em = pars['alpha_e']
    return -1*GF*a_em/np.pi*mlepton[lepton]*mK0*fK0*ckm_xi('t', 'sd')*(C10sdRe+1j*C10sdIm)

def amp_K0_leptons_LD(lepton: str) -> complex:
    a_em = alpha_em(mK0)
    #a_em = pars['alpha_e']
    aLD = amplitudes_LD(pars, 'K0', lepton)
    return GF*a_em*mK0**2*fK0/np.pi/np.sqrt(2)*np.array(aLD)

def amp_K0_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    if ma > couplings.ew_scale:
        basis = 'massbasis_above'
    else:
        basis = 'kF_below'
    cc = couplings.match_run(ma, basis, **kwargs)
    clep = cc['ke'][genlepton[lepton],genlepton[lepton]] - cc['kE'][genlepton[lepton],genlepton[lepton]]
    csd = cc['kD'][1,0] - cc['kd'][1,0]
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return - csd*clep/fa**2 *fK0*mlepton[lepton]/np.sqrt(2)*mK0**3/(mK0**2-ma**2+1j*ma*Gamma_a)

def amp_KL_leptons_P(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    amp_K0 = amp_K0_leptons_SM(lepton) + amp_K0_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_K0bar = np.conj(amp_K0)
    eps = epsilonKaon*(np.cos(phiepsilonKaon)+1j*np.sin(phiepsilonKaon))
    amp = ((1+eps)*amp_K0+(1-eps)*amp_K0bar)/np.sqrt(2*(1+np.abs(eps)**2))
    return amp + amp_K0_leptons_LD(lepton)[1]

def amp_KS_leptons_P(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> complex:
    amp_K0 = amp_K0_leptons_SM(lepton) + amp_K0_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_K0bar = np.conj(amp_K0)
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
    amp_S = amp_K0_leptons_LD(lepton)[0]
    amp_sq = np.abs(amp_P)**2 + np.abs(amp_S)**2 * (1-4*mlepton[lepton]**2/mKS**2)
    gamma = amp_sq*np.sqrt(kallen(mKS**2, mlepton[lepton]**2, mlepton[lepton]**2))/(16*np.pi*mKS**3)
    return gamma/GammaKS
from ...constants import GF, mB0, mBs, fB, fBs, C10, me, mmu, mtau, GammaB, GammaBs, DeltaGamma_Bs
from ...common import kallen, ckm_xi, alpha_em
from ...rge.classes import ALPcouplings
from ..alp_decays.branching_ratios import total_decay_width
import numpy as np

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
    cbs = - cc['kD'][2,1]
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
    cbs = - cc['kD'][2,0]
    Gamma_a = total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
    return - cbs*clep/fa**2 *fB*mlepton[lepton]/np.sqrt(2)*mB0**3/(mB0**2-ma**2+1j*ma*Gamma_a)

def BR_Bd_leptons_ALP(lepton: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float, **kwargs) -> float:
    amp_SM = amp_Bd_leptons_SM(lepton)
    amp_ALP = amp_Bd_leptons_ALP(lepton, ma, couplings, fa, br_dark, **kwargs)
    amp_sq = np.abs(amp_SM + amp_ALP)**2
    gamma_th = amp_sq*np.sqrt(kallen(mB0**2, mlepton[lepton]**2, mlepton[lepton]**2))/(16*np.pi*mB0**3)
    return gamma_th/GammaBs #The mixing correction is negligible for Bd
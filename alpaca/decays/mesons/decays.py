from .invisible import *
from .visible import *
from ...constants import(
    mUpsilon1S, BeeUpsilon1S,
    mUpsilon3S,
    mUpsilon4S, BeeUpsilon4S,
    mJpsi, BeeJpsi,
    GammaB, GammaB0,
    GammaJpsi, GammaUpsilon1S, GammaUpsilon3S, GammaUpsilon4S,
    GammaD0, GammaDplus, GammaDs,
    GammaK, GammaKL, GammaKS,
)
from ..nwa import transition_nwa
from ..alp_decays.branching_ratios import decay_channels
import numpy as np

meson_to_alp = {
    ('Upsilon(1S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, GammaUpsilon1S, 'b', fa, **kwargs),
    ('Upsilon(3S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs),
    ('Upsilon(4S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mUpsilon4S, BeeUpsilon4S, GammaUpsilon4S, 'b', fa, **kwargs),
    ('J/psi', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mJpsi, BeeJpsi, GammaJpsi, 'c', fa, **kwargs),
    ('B+', ('K+', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: brBplusKa(ma, couplings, fa, **kwargs),
    ('B-', ('K-', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: brBplusKa(ma, couplings, fa, **kwargs),
    ('B0', ('K0', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: brB0Ka(ma, couplings, fa, **kwargs),
    ('B0', ('K*0', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: B0toKsta(ma, couplings, fa, **kwargs)/GammaB0,
    ('B+', ('K*+', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: BplustoKsta(ma, couplings, fa, **kwargs)/GammaB,
    ('B-', ('K*-', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: BplustoKsta(ma, couplings, fa, **kwargs)/GammaB,
    ('B+', ('alp', 'pion+')): lambda ma, couplings, fa, br_dark, **kwargs: Btopia(ma, couplings, fa, **kwargs)/GammaB,
    ('B-', ('alp', 'pion-')): lambda ma, couplings, fa, br_dark, **kwargs: Btopia(ma, couplings, fa, **kwargs)/GammaB,
    ('B0', ('alp', 'pion0')): lambda ma, couplings, fa, br_dark, **kwargs: B0topia(ma, couplings, fa, **kwargs)/GammaB0,
    ('B0', ('alp', 'rho0')): lambda ma, couplings, fa, br_dark, **kwargs: B0torhoa(ma, couplings, fa, **kwargs)/GammaB0,
    ('B+', ('alp', 'rho+')): lambda ma, couplings, fa, br_dark, **kwargs: Bplustorhoa(ma, couplings, fa, **kwargs)/GammaB,
    ('B-', ('alp', 'rho-')): lambda ma, couplings, fa, br_dark, **kwargs: Bplustorhoa(ma, couplings, fa, **kwargs)/GammaB,
    ('Bs', ('alp', 'phi')): lambda ma, couplings, fa, br_dark, **kwargs: Bstophia(ma, couplings, fa, **kwargs)/GammaBs,
    ('K+', ('alp', 'pion+')): lambda ma, couplings, fa, br_dark, **kwargs: Kplustopia(ma, couplings, fa, **kwargs),
    ('K-', ('alp', 'pion-')): lambda ma, couplings, fa, br_dark, **kwargs: Kplustopia(ma, couplings, fa, **kwargs),
    ('KL', ('alp', 'pion0')): lambda ma, couplings, fa, br_dark, **kwargs: KLtopia(ma, couplings, fa, **kwargs),
    ('KS', ('alp', 'pion0')): lambda ma, couplings, fa, br_dark, **kwargs: KStopia(ma, couplings, fa, **kwargs),
    ('D0', ('alp', 'pion0')): lambda ma, couplings, fa, br_dark, **kwargs: D0topi0a(ma, couplings, fa, **kwargs)/GammaD0,
    ('D0', ('alp', 'eta')): lambda ma, couplings, fa, br_dark, **kwargs: D0toetaa(ma, couplings, fa, **kwargs)/GammaD0,
    ('D0', ('alp', 'eta_prime')): lambda ma, couplings, fa, br_dark, **kwargs: D0toetapa(ma, couplings, fa, **kwargs)/GammaD0,
    ('D0', ('alp', 'rho0')): lambda ma, couplings, fa, br_dark, **kwargs: D0torhoa(ma, couplings, fa, **kwargs)/GammaD0,
    ('D+', ('alp', 'pion+')): lambda ma, couplings, fa, br_dark, **kwargs: Dplustopiplusa(ma, couplings, fa, **kwargs)/GammaDplus,
    ('D-', ('alp', 'pion-')): lambda ma, couplings, fa, br_dark, **kwargs: Dplustopiplusa(ma, couplings, fa, **kwargs)/GammaDplus,
    ('D+', ('alp', 'rho+')): lambda ma, couplings, fa, br_dark, **kwargs: Dplustorhoa(ma, couplings, fa, **kwargs)/GammaDplus,
    ('D-', ('alp', 'rho-')): lambda ma, couplings, fa, br_dark, **kwargs: Dplustorhoa(ma, couplings, fa, **kwargs)/GammaDplus,
    ('Ds+', ('K+', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: DstoKa(ma, couplings, fa, **kwargs)/GammaDs,
    ('Ds-', ('K-', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: DstoKa(ma, couplings, fa, **kwargs)/GammaDs,
    ('Ds+', ('K*+', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: DstoKsta(ma, couplings, fa, **kwargs)/GammaDs,
    ('Ds-', ('K*-', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: DstoKsta(ma, couplings, fa, **kwargs)/GammaDs,
}

meson_nwa = {}
for meson_process in meson_to_alp.keys():
    for channel in decay_channels:
        meson_nwa[transition_nwa(meson_process, channel)] = (meson_process, channel)

meson_mediated = {
    ('Bs', ('electron', 'electron')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bs_leptons_ALP('e', ma, couplings, fa, br_dark, **kwargs),
    ('Bs', ('muon', 'muon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bs_leptons_ALP('mu', ma, couplings, fa, br_dark, **kwargs),
    ('Bs', ('tau', 'tau')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bs_leptons_ALP('tau', ma, couplings, fa, br_dark, **kwargs),
    ('Bs', ('photon', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bs_photons_ALP(ma, couplings, fa, br_dark, **kwargs),
    ('B0', ('electron', 'electron')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bd_leptons_ALP('e', ma, couplings, fa, br_dark, **kwargs),
    ('B0', ('muon', 'muon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bd_leptons_ALP('mu', ma, couplings, fa, br_dark, **kwargs),
    ('B0', ('tau', 'tau')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bd_leptons_ALP('tau', ma, couplings, fa, br_dark, **kwargs),
    ('B0', ('photon', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Bd_photons_ALP(ma, couplings, fa, br_dark, **kwargs),
    ('D0', ('photon', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_D0_photons_ALP(ma, couplings, fa, br_dark, **kwargs),
    ('D0', ('electron', 'electron')): lambda ma, couplings, fa, br_dark, **kwargs: BR_D0_leptons_ALP('e', ma, couplings, fa, br_dark, **kwargs),
    ('D0', ('muon', 'muon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_D0_leptons_ALP('mu', ma, couplings, fa, br_dark, **kwargs),
    ('KL', ('electron', 'electron')): lambda ma, couplings, fa, br_dark, **kwargs: BR_KL_leptons('e', ma, couplings, fa, br_dark, **kwargs),
    ('KL', ('muon', 'muon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_KL_leptons('mu', ma, couplings, fa, br_dark, **kwargs),
    ('KS', ('electron', 'electron')): lambda ma, couplings, fa, br_dark, **kwargs: BR_KS_leptons('e', ma, couplings, fa, br_dark, **kwargs),
    ('KS', ('muon', 'muon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_KS_leptons('mu', ma, couplings, fa, br_dark, **kwargs),
    ('KS', ('photon', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_KS_photons(ma, couplings, fa, br_dark, **kwargs),
}

meson_widths = {
    'Upsilon(1S)': GammaUpsilon1S,
    'Upsilon(3S)': GammaUpsilon3S,
    'Upsilon(4S)': GammaUpsilon4S,
    'J/psi': GammaJpsi,
    'B+': GammaB,
    'B-': GammaB,
    'B0': GammaB0,
    'D0': GammaD0,
    'D+': GammaDplus,
    'Ds+': GammaDs,
    'K+': GammaK,
    'K-': GammaK,
    'KL': GammaKL,
    'KS': GammaKS,
}

def meson_width(meson, ma, couplings, fa, br_dark=0, **kwargs):
    """
    Calculate the total width of a meson decaying to ALP and other particles.
    
    Parameters:
    - meson: str, name of the meson
    - ma: float, mass of the ALP
    - couplings: dict, couplings relevant for the decay
    - fa: float, decay constant of the ALP
    - br_dark: float, branching ratio to dark sector (default 0)
    
    Returns:
    - float, total width of the meson decay
    """
    if meson not in meson_widths:
        raise ValueError(f"Meson {meson} not recognized.")
    
    return meson_widths[meson] * np.sum([v(ma, couplings, fa, br_dark, **kwargs) for k, v in meson_to_alp.items() if k[0] == meson])

from .invisible import *
from .visible import *
from ...constants import(
    mUpsilon1S, BeeUpsilon1S,
    mUpsilon3S,
    mUpsilon4S, BeeUpsilon4S,
    mJpsi, BeeJpsi,
    GammaB, GammaB0,
    GammaD0, GammaDplus, GammaDs
)
from ..nwa import transition_nwa
from ..alp_decays.branching_ratios import decay_channels

meson_to_alp = {
    ('Upsilon(1S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs),
    ('Upsilon(3S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs),
    ('Upsilon(4S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mUpsilon4S, BeeUpsilon4S, 'b', fa, **kwargs),
    ('J/psi', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mJpsi, BeeJpsi, 'c', fa, **kwargs),
    ('B+', ('K+', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: BtoKa(ma, couplings, fa, **kwargs)/GammaB,
    ('B-', ('K-', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: BtoKa(ma, couplings, fa, **kwargs)/GammaB,
    ('B0', ('K0', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: B0toKa(ma, couplings, fa, **kwargs)/GammaB0,
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
}
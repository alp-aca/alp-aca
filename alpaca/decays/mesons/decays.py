from .invisible import *
from ...constants import(
    mUpsilon1S, BeeUpsilon1S,
    mUpsilon3S,
    mUpsilon4S, BeeUpsilon4S,
    mJpsi, BeeJpsi,
    GammaB, GammaB0,
)
from ..alp_decays.branching_ratios import decay_channels, BRsalp

meson_to_alp = {
    ('Upsilon(1S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs),
    ('Upsilon(3S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs),
    ('Upsilon(4S)', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mUpsilon4S, BeeUpsilon4S, 'b', fa, **kwargs),
    ('J/psi', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: BR_Vagamma(ma, couplings, mJpsi, BeeJpsi, 'c', fa, **kwargs),
    ('B+', ('K+', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: BtoKa(ma, couplings, fa, **kwargs)/GammaB,
    ('B0', ('K0', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: B0toKa(ma, couplings, fa, **kwargs)/GammaB0,
    ('B0', ('K*0', 'alp')): lambda ma, couplings, fa, br_dark, **kwargs: B0toKsta(ma, couplings, fa, **kwargs)/GammaB0,
    ('K+', ('alp', 'pion+')): lambda ma, couplings, fa, br_dark, **kwargs: Kplustopia(ma, couplings, fa, **kwargs),
    ('KL', ('alp', 'pion0')): lambda ma, couplings, fa, br_dark, **kwargs: KLtopia(ma, couplings, fa, **kwargs),
}

def transition_nwa(
        production_channel: tuple[tuple[str], tuple[str]],
        decay_channel: tuple[str],) -> tuple[str, tuple[str]]:
    final = [particle for particle in production_channel[1] if particle != 'alp']
    final += decay_channel
    return (production_channel[0], tuple(sorted(final)))

meson_nwa = {}
for meson_process in meson_to_alp.keys():
    for channel in decay_channels:
        meson_nwa[transition_nwa(meson_process, channel)] = (meson_process, channel)
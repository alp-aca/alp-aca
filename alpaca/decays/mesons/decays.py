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
    ('Upsilon(1S)', ('alp', 'photon')): lambda ma, couplings, fa, **kwargs: BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs),
    ('Upsilon(3S)', ('alp', 'photon')): lambda ma, couplings, fa, **kwargs: Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs),
    ('Upsilon(4S)', ('alp', 'photon')): lambda ma, couplings, fa, **kwargs: BR_Vagamma(ma, couplings, mUpsilon4S, BeeUpsilon4S, 'b', fa, **kwargs),
    ('J/psi', ('alp', 'photon')): lambda ma, couplings, fa, **kwargs: BR_Vagamma(ma, couplings, mJpsi, BeeJpsi, 'c', fa, **kwargs),
    ('B+', ('alp', 'K+')): lambda ma, couplings, fa, **kwargs: BtoKa(ma, couplings, fa, **kwargs)/GammaB,
    ('B0', ('alp', 'K0')): lambda ma, couplings, fa, **kwargs: B0toKa(ma, couplings, fa, **kwargs)/GammaB0,
    ('B0', ('alp', 'K*0')): lambda ma, couplings, fa, **kwargs: B0toKsta(ma, couplings, fa, **kwargs)/GammaB0,
    ('K+', ('alp', 'pion+')): Kplustopia,
    ('KL', ('alp', 'pion0')): KLtopia,
}

def transition_nwa(
        production_channel: tuple[list[str], list[str]],
        decay_channel: list[str],) -> tuple[list[str], list[str]]:
    final = sorted(set(production_channel[1]+decay_channel)-set(['alp']))
    return (production_channel[0], final)

meson_nwa = {}
for meson_process, meson_br in meson_to_alp.items():
    for channel in decay_channels:
        meson_nwa[transition_nwa(meson_process, channel)] = lambda ma, couplings, fa, br_dark, **kwargs: meson_br(ma, couplings, fa, **kwargs) * BRsalp(ma, couplings, fa, br_dark, **kwargs)[channel]
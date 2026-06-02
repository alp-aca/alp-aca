from .Z import decaywidth_Z_to_agamma
from ..nwa import transition_nwa
from ...constants import GammaZ
from ..alp_decays.branching_ratios import decay_channels

gauge_to_alp = {
    ('Z', ('alp', 'photon')): lambda ma, couplings, fa, br_dark, **kwargs: decaywidth_Z_to_agamma(ma, couplings, fa)/GammaZ,
}

gauge_nwa = {}
for gauge_process in gauge_to_alp.keys():
    for channel in decay_channels:
        gauge_nwa[transition_nwa(gauge_process, channel)] = (gauge_process, channel)
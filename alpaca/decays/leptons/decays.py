from ...constants import Gammatau, Gammamu
from .lfv import tau_mua, tau_ea, mu_ea
from ..nwa import transition_nwa
from ..alp_decays.branching_ratios import decay_channels

lepton_to_alp = {
    ('tau', ('alp', 'muon')): lambda ma, couplings, fa, br_dark, **kwargs: tau_mua(ma, couplings, fa, **kwargs)/Gammatau,
    ('tau', ('alp', 'electron')): lambda ma, couplings, fa, br_dark, **kwargs: tau_ea(ma, couplings, fa, **kwargs)/Gammatau,
    ('muon', ('alp', 'electron')): lambda ma, couplings, fa, br_dark, **kwargs: mu_ea(ma, couplings, fa, **kwargs)/Gammamu,
}

lepton_nwa = {}
for lepton_process in lepton_to_alp.keys():
    for channel in decay_channels:
        lepton_nwa[transition_nwa(lepton_process, channel)] = (lepton_process, channel)
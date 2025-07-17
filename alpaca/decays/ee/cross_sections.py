import numpy as np
from ...biblio.biblio import citations
from ...rge.classes import ALPcouplings
from ..nwa import transition_nwa
from ..alp_decays.branching_ratios import decay_channels
from ..effcouplings import offshellphoton

def sigmaNR_gammaALP(ma: float, couplings: ALPcouplings, s: float, f_a: float=1000,**kwargs):
    citations.register_inspire('Merlo:2019anv')
    citations.register_inspire('DiLuzio:2024jip')
    from ...constants import hbarc2_GeV2pb
    from ...common import alpha_em
    coup_low = couplings.match_run(ma, 'RL_below', **kwargs)
    gaphoton = offshellphoton(coup_low, ma, s)*alpha_em(np.sqrt(s))/(np.pi*f_a)
    return hbarc2_GeV2pb*(((alpha_em(np.sqrt(s))*np.abs(gaphoton)**2)/24)*(1-(ma**2)/s)**3)

xsections = {
    (('electron', 'electron'), ('alp', 'photon')): lambda ma, couplings, s, fa, br_dark, **kwargs: sigmaNR_gammaALP(ma, couplings, s, fa, **kwargs),
}

xsections_nwa = {}
for xsection_process in xsections.keys():
    for channel in decay_channels:
        xsections_nwa[transition_nwa(xsection_process, channel)] = (xsection_process, channel)
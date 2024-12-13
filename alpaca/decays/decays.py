from .mesons import invisible
from .alp_decays import hadronic_decays_def, gaugebosons, fermion_decays, branching_ratios
from ..rge import ALPcouplings
from .particles import particle_aliases
import numpy as np

def parse(transition: str) -> tuple[list[str], list[str]]:
    initial, final = transition.split('->')
    initial = sorted([particle_aliases[p.strip()] for p in initial.split()])
    final = sorted([particle_aliases[p.strip()] for p in final.split()])
    return initial, final

def decay_width(transition: str, ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    initial, final = parse(transition)
    if initial == ['alp'] and final == ['electron', 'electron']:
        dw = fermion_decays.decay_width_electron
    elif initial == ['alp'] and final == ['muon', 'muon']:
        dw = fermion_decays.decay_width_muon
    elif initial == ['alp'] and final == ['tau', 'tau']:
        dw = fermion_decays.decay_width_tau
    elif initial == ['alp'] and final == ['photon', 'photon']:
        dw = gaugebosons.decay_width_2gamma
    elif initial == ['alp'] and final == sorted(['eta', 'pion0', 'pion0']):
        dw = hadronic_decays_def.decay_width_etapipi00
    elif initial == ['alp'] and final == sorted(['eta', 'pion+', 'pion-']):
        dw = hadronic_decays_def.decay_width_etapipipm
    elif initial == ['alp'] and final == ['eta', 'pion', 'pion']:
        dw = lambda ma, couplings, fa, **kwargs: hadronic_decays_def.decay_width_etapipi00(ma, couplings, fa, **kwargs) + hadronic_decays_def.decay_width_etapipipm(ma, couplings, fa, **kwargs)
    elif initial == ['alp'] and (final == sorted(['photon', 'pion', 'pion']) or final == sorted(['photon', 'pion+', 'pion-'])):
        dw = hadronic_decays_def.decay_width_gammapipi
    else:
        raise NotImplementedError(f'Unknown decay process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(dw)(ma, couplings, fa, **kwargs)

def branching_ratio(transition: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float = 0, **kwargs) -> float:
    initial, final = parse(transition)
    #Initial resonance Upsilon (1S)
    if initial == ['Upsilon(1S)'] and final == sorted(['photon', 'alp']):
        from ..constants import mUpsilon1S, BeeUpsilon1S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs)
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'muon', 'muon']):
        from ..constants import mUpsilon1S, BeeUpsilon1S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'tau', 'tau']):
        from ..constants import mUpsilon1S, BeeUpsilon1S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['tau']
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'charm', 'charm']):
        from ..constants import mUpsilon1S, BeeUpsilon1S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['charm']
    #Initial resonance Upsilon 3S
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'alp']):
        from ..constants import mUpsilon3S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs)
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'hadrons']):
        from ..constants import mUpsilon3S, BeeUpsilon3S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['hadrons']
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'muon', 'muon']):
        from ..constants import mUpsilon3S, BeeUpsilon3S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'tau', 'tau']):
        from ..constants import mUpsilon3S, BeeUpsilon3S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Mixed_QuarkoniaSearches(ma, couplings, mUpsilon3S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['tau']
    #Initial resonance Upsilon 4S
    elif initial == ['Upsilon(4S)'] and final == sorted(['photon', 'alp']):
        from ..constants import mUpsilon4S, BeeUpsilon4S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon4S, BeeUpsilon4S, 'b', fa, **kwargs) 
    elif initial == ['Upsilon(4S)'] and final == sorted(['photon', 'photon', 'photon']):
        from ..constants import mUpsilon4S, BeeUpsilon4S
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon4S, BeeUpsilon4S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    #Initial resonance J/psi    
    elif initial == ['J/psi'] and final == sorted(['photon', 'alp']):
        from ..constants import mJpsi, BeeJpsi
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mJpsi, BeeJpsi, 'c', fa, **kwargs)
    elif initial == ['J/psi'] and final == sorted(['photon', 'muon', 'muon']):
        from ..constants import mJpsi
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Mixed_QuarkoniaSearches(ma, couplings, mJpsi, 'c', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['J/psi'] and final == sorted(['photon', 'photon', 'photon']):
        from ..constants import mJpsi, BeeJpsi
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BR_Vagamma(ma, couplings, mJpsi, BeeJpsi, 'c', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    elif initial == ['B+'] and final == sorted(['alp', 'K+']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB
    elif initial == ['B0'] and final == sorted(['alp', 'K*0']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0
    elif initial == ['B+'] and final == sorted(['K+', 'muon', 'muon']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['B0'] and final == sorted(['K*0', 'muon', 'muon']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['B+'] and final == sorted(['K+', 'electron', 'electron']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['e']
    elif initial == ['B0'] and final == sorted(['K*0', 'electron', 'electron']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['e']
    elif initial == ['B+'] and final == sorted(['K+', 'tau', 'tau']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['tau']
    elif initial == ['B0'] and final == sorted(['K*0', 'tau', 'tau']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['tau']
    elif initial == ['B+'] and final == sorted(['K+', 'photon', 'photon']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    elif initial == ['B0'] and final == sorted(['K*0', 'photon', 'photon']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    
    else:
        raise NotImplementedError(f'Unknown branching ratio process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(br)(ma, couplings, fa, br_dark, **kwargs)

def cross_section(transition: str, ma: float, couplings: ALPcouplings, s: float, fa: float, br_dark=0, **kwargs) -> float:
    initial, final = parse(transition)
    if initial == ['electron', 'electron'] and final == sorted(['alp', 'photon']):
        sigma = invisible.sigmaNR
    elif initial == ['electron', 'electron'] and final == sorted(['photon', 'photon', 'photon']):
        sigma = lambda ma, couplings, s, fa, br_dark, **kwargs: invisible.sigmaNR(ma, couplings, s, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    else:
        raise NotImplementedError(f'Unknown cross section process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(sigma)(ma, couplings, s, fa, br_dark, **kwargs)
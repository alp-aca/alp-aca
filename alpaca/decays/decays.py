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
    elif initial == ['alp'] and final == ['charm', 'charm']:
        dw = fermion_decays.decay_width_charm
    elif initial == ['alp'] and final == ['bottom', 'bottom']:
        dw = fermion_decays.decay_width_bottom
    elif initial == ['alp'] and final == ['photon', 'photon']:
        dw = gaugebosons.decay_width_2gamma
    elif initial == ['alp'] and final == ['gluon', 'gluon']:
        dw = gaugebosons.decay_width_2gluons
    elif initial == ['alp'] and final == ['hadrons']:
        dw = hadronic_decays_def.decay_width_hadrons
    elif initial == ['alp'] and final == sorted(['pion0', 'pion0', 'pion0']):
        dw = hadronic_decays_def.decay_width_3pi000
    elif initial == ['alp'] and final == sorted(['pion0', 'pion+', 'pion-']):
        dw = hadronic_decays_def.decay_width_3pi0pm
    elif initial == ['alp'] and final == sorted(['pion', 'pion', 'pion']):
        dw = lambda ma, couplings, fa, **kwargs: hadronic_decays_def.decay_width_3pi000(ma, couplings, fa, **kwargs) + hadronic_decays_def.decay_width_3pi0pm(ma, couplings, fa, **kwargs)
    elif initial == ['alp'] and final == sorted(['eta', 'pion0', 'pion0']):
        dw = hadronic_decays_def.decay_width_etapipi00
    elif initial == ['alp'] and final == sorted(['eta', 'pion+', 'pion-']):
        dw = hadronic_decays_def.decay_width_etapipipm
    elif initial == ['alp'] and final == ['eta', 'pion', 'pion']:
        dw = lambda ma, couplings, fa, **kwargs: hadronic_decays_def.decay_width_etapipi00(ma, couplings, fa, **kwargs) + hadronic_decays_def.decay_width_etapipipm(ma, couplings, fa, **kwargs)
    elif initial == ['alp'] and final == sorted(['eta_prime', 'pion0', 'pion0']):
        dw = hadronic_decays_def.decay_width_etappipi00
    elif initial == ['alp'] and final == sorted(['eta_prime', 'pion+', 'pion-']):
        dw = hadronic_decays_def.decay_width_etappipipm
    elif initial == ['alp'] and final == ['eta_prime', 'pion', 'pion']:
        dw = lambda ma, couplings, fa, **kwargs: hadronic_decays_def.decay_width_etappipi00(ma, couplings, fa, **kwargs) + hadronic_decays_def.decay_width_etappipipm(ma, couplings, fa, **kwargs)
    elif initial == ['alp'] and (final == sorted(['photon', 'pion', 'pion']) or final == sorted(['photon', 'pion+', 'pion-'])):
        dw = hadronic_decays_def.decay_width_gammapipi
    elif initial == ['alp'] and final == sorted(['omega', 'omega']):
        dw = hadronic_decays_def.decay_width_2w
    else:
        raise NotImplementedError(f'Unknown decay process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(dw)(ma, couplings, fa, **kwargs)

def branching_ratio(transition: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float = 0, **kwargs) -> float:
    initial, final = parse(transition)
    # ALP decays
    if initial == ['alp'] and final == ['electron', 'electron']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['e']
    elif initial == ['alp'] and final == ['muon', 'muon']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['alp'] and final == ['tau', 'tau']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['tau']
    elif initial == ['alp'] and final == ['charm', 'charm']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['charm']
    elif initial == ['alp'] and final == ['bottom', 'bottom']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['bottom']
    elif initial == ['alp'] and final == ['photon', 'photon']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    elif initial == ['alp'] and final == ['gluon', 'gluon']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['gluongluon']
    elif initial == ['alp'] and final == ['hadrons']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['hadrons']
    elif initial == ['alp'] and final == sorted(['pion0', 'pion0', 'pion0']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['pi0pi0pi0']
    elif initial == ['alp'] and final == sorted(['pion0', 'pion+', 'pion-']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['pi0pippim']
    elif initial == ['alp'] and final == sorted(['pion', 'pion', 'pion']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['3pis']
    elif initial == ['alp'] and final == sorted(['eta', 'pion0', 'pion0']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['etapi0pi0']
    elif initial == ['alp'] and final == sorted(['eta', 'pion+', 'pion-']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['etapippim']
    elif initial == ['alp'] and final == sorted(['eta', 'pion', 'pion']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['etapipi']
    elif initial == ['alp'] and final == sorted(['eta_prime', 'pion0', 'pion0']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['etappi0pi0']
    elif initial == ['alp'] and final == sorted(['eta_prime', 'pion+', 'pion-']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['etappippim']
    elif initial == ['alp'] and final == sorted(['eta_prime', 'pion', 'pion']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['etappipi']
    elif initial == ['alp'] and final == sorted(['photon', 'pion', 'pion']) or final == sorted(['photon', 'pion+', 'pion-']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['gammapipi']
    elif initial == ['alp'] and final == sorted(['omega', 'omega']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2omega']
    elif initial == ['alp'] and final == ['dark']:
        br = lambda ma, couplings, fa, br_dark, **kwargs: br_dark
    #Initial resonance Upsilon (1S)
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'alp']):
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
    #Initial B meson
    elif initial == ['B+'] and final == sorted(['alp', 'K+']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB
    elif initial == ['B+'] and final == sorted(['K+', 'muon', 'muon']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['B+'] and final == sorted(['K+', 'electron', 'electron']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['e']
    elif initial == ['B+'] and final == sorted(['K+', 'tau', 'tau']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['tau']
    elif initial == ['B+'] and final == sorted(['K+', 'photon', 'photon']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    elif initial == ['B+'] and final == sorted(['K+', 'pion+', 'pion-', 'pion0']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['pi0pippim']
    elif initial == ['B+'] and final == sorted(['K+', 'eta', 'pion+', 'pion-']):
        from ..constants import GammaB
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.BtoKa(ma, couplings, fa, **kwargs)/GammaB * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['etapippim']
    elif initial == ['B0'] and final == sorted(['alp', 'K*0']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0
    elif initial == ['B0'] and final == sorted(['K*0', 'muon', 'muon']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['B0'] and final == sorted(['K*0', 'electron', 'electron']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['e']
    elif initial == ['B0'] and final == sorted(['K*0', 'tau', 'tau']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['tau']
    elif initial == ['B0'] and final == sorted(['K*0', 'photon', 'photon']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKsta(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    elif initial == ['B0'] and final == sorted(['K0', 'pion+', 'pion-', 'pion0']):
        from ..constants import GammaB0
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.B0toKa(ma, couplings, fa, **kwargs)/GammaB0 * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['pi0pippim']
    #Initial K meson
    elif initial == ['K+'] and final == sorted(['alp', 'pion+']):
        br = invisible.Kplustopia
    elif initial == ['K+'] and final == sorted(['pion+', 'photon', 'photon']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Kplustopia(ma, couplings, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    elif initial == ['K+'] and final == sorted(['pion+', 'muon', 'muon']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Kplustopia(ma, couplings, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['K+'] and final == sorted(['pion+', 'electron', 'electron']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.Kplustopia(ma, couplings, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['e']
    elif initial == ['KL'] and final == sorted(['alp', 'pion0']):
        br = invisible.KLtopia
    elif initial == ['KL'] and final == sorted(['pion0', 'photon', 'photon']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.KLtopia(ma, couplings, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    elif initial == ['KL'] and final == sorted(['pion0', 'muon', 'muon']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.KLtopia(ma, couplings, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['mu']
    elif initial == ['KL'] and final == sorted(['pion0', 'electron', 'electron']):
        br = lambda ma, couplings, fa, br_dark, **kwargs: invisible.KLtopia(ma, couplings, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['e']
    
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
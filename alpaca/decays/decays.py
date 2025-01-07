from .mesons import invisible
from .alp_decays import hadronic_decays_def, gaugebosons, fermion_decays, branching_ratios
from ..rge import ALPcouplings
from .particles import particle_aliases
from .mesons.decays import meson_to_alp, meson_nwa
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
    if initial == ['alp'] and tuple(final) in branching_ratios.decay_channels:
        br = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)[tuple(final)]
    # Meson decays to ALP
    elif len(initial) == 1 and (initial[0], tuple(final)) in meson_to_alp.keys():
        br = meson_to_alp[(initial[0], tuple(final))]
    # Meson decays in NWA
    elif len(initial) == 1 and (initial[0], tuple(final)) in meson_nwa.keys():
        meson_process, channel = meson_nwa[(initial[0], tuple(final))]
        br = lambda ma, couplings, fa, br_dark, **kwargs: meson_to_alp[meson_process](ma, couplings, fa, br_dark, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)[channel]
    else:
        raise NotImplementedError(f'Unknown branching ratio process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(br, otypes=[float])(ma, couplings, fa, br_dark, **kwargs)

def cross_section(transition: str, ma: float, couplings: ALPcouplings, s: float, fa: float, br_dark=0, **kwargs) -> float:
    initial, final = parse(transition)
    if initial == ['electron', 'electron'] and final == sorted(['alp', 'photon']):
        sigma = invisible.sigmaNR
    elif initial == ['electron', 'electron'] and final == sorted(['photon', 'photon', 'photon']):
        sigma = lambda ma, couplings, s, fa, br_dark, **kwargs: invisible.sigmaNR(ma, couplings, s, fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)['2photons']
    else:
        raise NotImplementedError(f'Unknown cross section process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(sigma)(ma, couplings, s, fa, br_dark, **kwargs)
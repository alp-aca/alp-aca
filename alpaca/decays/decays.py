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

def branching_ratio(transition: str, ma: float, couplings: ALPcouplings, fa: float, **kwargs) -> float:
    initial, final = parse(transition)
    if initial == ['Upsilon(1S)'] and final == sorted(['photon', 'muon', 'muon']):
        from ..constants import mUpsilon1S, BeeUpsilon1S
        br = lambda ma, couplings, fa, **kwargs: invisible.BR_Vagamma(ma, couplings, mUpsilon1S, BeeUpsilon1S, 'b', fa, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, **kwargs)['mu']
    else:
        raise NotImplementedError(f'Unknown branching ratio process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(br)(ma, couplings, fa, **kwargs)

def cross_section(transition: str, ma: float, couplings: ALPcouplings, s: float, fa: float, **kwargs) -> float:
    initial, final = parse(transition)
    if initial == ['electron', 'electron'] and final == sorted(['alp', 'photon']):
        sigma = invisible.sigmaNR
    else:
        raise NotImplementedError(f'Unknown cross section process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(sigma)(ma, couplings, s, fa, **kwargs)
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

def decay_width(transition: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float = 0.0, **kwargs) -> float:
    """ Calculate the decay width for a given transition.

    Parameters
    ----------
    transition (str) : 
        The particle transition in the form 'initial -> final'.
    ma (float) :
        The mass of the ALP.
    couplings (ALPcouplings) :
        The couplings of the ALP to other particles.
    fa (float):
        The decay constant of the ALP.
    br_dark (float, optional):
        The branching ratio to dark sector particles. Default is 0.0.
    **kwargs:
        Additional parameters for the decay width calculation.

    Returns
    -------
    Gamma (float) :
        The decay width for the specified transition, in GeV.

    Raises
    ------
        NotImplementedError: If the decay process is unknown.
    """
    if particle_aliases.get(transition.strip()) == 'alp':
        dw = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot']
        return np.vectorize(dw)(ma, couplings, fa, br_dark, **kwargs)
    initial, final = parse(transition)
    # ALP decays
    if initial == ['alp']:
        dw = lambda ma, couplings, fa, br_dark, **kwargs: branching_ratios.total_decay_width(ma, couplings, fa, br_dark, **kwargs)['DW_tot'] * branching_ratio(transition, ma, couplings, fa, br_dark, **kwargs)
    else:
        raise NotImplementedError(f'Unknown decay process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(dw)(ma, couplings, fa, br_dark, **kwargs)

def branching_ratio(transition: str, ma: float, couplings: ALPcouplings, fa: float, br_dark: float = 0.0, **kwargs) -> float:
    """ Calculate the branching ratio for a given transition.

    Parameters
    ----------
    transition (str) : 
        The particle transition in the form 'initial -> final'.
    ma (float) :
        The mass of the ALP.
    couplings (ALPcouplings) :
        The couplings of the ALP to other particles.
    fa (float):
        The decay constant of the ALP.
    br_dark (float, optional):
        The branching ratio to dark sector particles. Default is 0.0.
    **kwargs:
        Additional parameters for the branching ratio calculation.

    Returns
    -------
    BR (float) :
        The branching ratio for the specified transition.

    Raises
    ------
        NotImplementedError: If the decay process is unknown.
    """
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
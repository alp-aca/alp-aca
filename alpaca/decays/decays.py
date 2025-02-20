from .mesons import invisible
from .alp_decays import branching_ratios
from ..rge import ALPcouplings
from .particles import particle_aliases
from .mesons.decays import meson_to_alp, meson_nwa, meson_mediated
from .leptons.decays import lepton_to_alp, lepton_nwa
from .ee.cross_sections import xsections as xsections_ee, xsections_nwa as xsections_nwa_ee
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
        The mass of the ALP, in GeV.
    couplings (ALPcouplings) :
        The couplings of the ALP to other particles.
    fa (float):
        The decay constant of the ALP, in GeV.
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
        The mass of the ALP, in GeV.
    couplings (ALPcouplings) :
        The couplings of the ALP to other particles.
    fa (float):
        The decay constant of the ALP, in GeV.
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
    # Meson decays mediated by off-shell ALP
    elif len(initial) == 1 and (initial[0], tuple(final)) in meson_mediated.keys():
        br = meson_mediated[(initial[0], tuple(final))]
    # Lepton decays to ALP
    elif len(initial) == 1 and (initial[0], tuple(final)) in lepton_to_alp.keys():
        br = lepton_to_alp[(initial[0], tuple(final))]
    # Lepton decays in NWA
    elif len(initial) == 1 and (initial[0], tuple(final)) in lepton_nwa.keys():
        lepton_process, channel = lepton_nwa[(initial[0], tuple(final))]
        br = lambda ma, couplings, fa, br_dark, **kwargs: lepton_to_alp[lepton_process](ma, couplings, fa, br_dark, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)[channel]
    else:
        raise NotImplementedError(f'Unknown branching ratio process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(br, otypes=[float])(ma, couplings, fa, br_dark, **kwargs)

def cross_section(transition: str, ma: float, couplings: ALPcouplings, s: float, fa: float, br_dark=0, **kwargs) -> float:
    """Calculate the cross section for a given transition process involving an ALP

    Parameters
    ----------
    transition (str) :
        The transition process in the form 'initial -> final'.
    ma (float) :
        The mass of the ALP, in GeV.
    couplings (ALPcouplings) :
        The couplings of the ALP to other particles.
    s (float) :
        The Mandelstam variable s, representing the square of the center-of-mass energy, in Gev^2.
    fa (float) :
        The decay constant of the ALP, in GeV.
    br_dark (float, optional) :
        The branching ratio to dark sector particles. Default is 0.
    **kwargs:
        Additional keyword arguments for specific cross section calculations.

    Returns
    -------
    sigma (float) :
        The calculated cross section for the given transition process.

    Raises
    ------
    NotImplementedError: If the transition process is not recognized or implemented.
    """
    initial, final = parse(transition)
    # ee -> alp + X
    if (tuple(initial), tuple(final)) in xsections_ee.keys():
        sigma = xsections_ee[(tuple(initial), tuple(final))]
    # ee -> alp + X -> final in NWA
    elif (tuple(initial), tuple(final)) in xsections_nwa_ee.keys():
        production, decay = xsections_nwa_ee[(tuple(initial), tuple(final))]
        sigma = lambda ma, couplings, s, fa, br_dark, **kwargs: xsections_ee[production](ma, couplings, s, fa, br_dark, **kwargs) * branching_ratios.BRsalp(ma, couplings, fa, br_dark=br_dark, **kwargs)[decay]
    else:
        raise NotImplementedError(f'Unknown cross section process {" ".join(initial)} -> {" ".join(final)}')
    
    return np.vectorize(sigma, otypes=[float])(ma, couplings, s, fa, br_dark, **kwargs)
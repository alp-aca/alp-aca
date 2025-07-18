import numpy as np
from ...rge import ALPcouplings
from .fermion_decays import decay_width_electron, decay_width_muon, decay_width_tau, decay_width_charm, decay_width_bottom, decay_width_etau, decay_width_mutau, decay_width_mue
from .hadronic_decays_def import decay_width_3pi000, decay_width_3pi0pm, decay_width_etapipi00, decay_width_etapipipm, decay_width_etappipi00, decay_width_etappipipm, decay_width_gammapipi, decay_width_2w
from .gaugebosons import decay_width_2gamma, decay_width_2gluons
from functools import cache

decay_channels =[
    ('electron', 'electron'),
    ('muon', 'muon'),
    ('tau', 'tau'),
    ('electron', 'muon'),
    ('electron', 'tau'),
    ('muon', 'tau'),
    ('charm', 'charm'),
    ('bottom', 'bottom'),
    ('pion', 'pion', 'pion'),
    ('pion+', 'pion-', 'pion0'),
    ('pion0', 'pion0', 'pion0'),
    ('eta', 'pion', 'pion'),
    ('eta', 'pion0', 'pion0'),
    ('eta', 'pion+', 'pion-'),
    ('eta_prime', 'pion', 'pion'),
    ('eta_prime', 'pion0', 'pion0'),
    ('eta_prime', 'pion+', 'pion-'),
    ('photon', 'pion', 'pion'),
    ('omega', 'omega'),
    ('gluon', 'gluon'),
    ('photon', 'photon'),
    ('hadrons',),
    ('dark',),
]

@cache
def _total_decay_width (ma, couplings: ALPcouplings, fa, br_dark = 0.0, **kwargs):
    if br_dark < 0.0 or br_dark > 1.0:
        raise ValueError('br_dark must be between in the interval [0,1]')
    if br_dark == 1.0:
        return {'e': 0.0, 'mu': 0.0, 'tau': 0.0, 'charm': 0.0, 'bottom': 0.0, '3pis': 0.0, 'etapipi': 0.0, 'etappipi': 0.0, 'gammapipi': 0.0, '2omega': 0.0, 'gluongluon': 0.0, '2photons': 0.0, 'DW_SM': 0.0, 'DW_dark': 1e30, 'DW_tot': 1e30}
    kwargs_nointegral = {k: v for k, v in kwargs.items() if k not in ['nitn_adapt', 'neval_adapt', 'nitn', 'neval', 'cores']}
    DW_elec = decay_width_electron(ma, couplings, fa, **kwargs_nointegral)
    DW_muon = decay_width_muon(ma, couplings, fa, **kwargs_nointegral)
    DW_tau = decay_width_tau(ma, couplings, fa, **kwargs_nointegral)
    DW_emu = decay_width_mue(ma, couplings, fa, **kwargs_nointegral)
    DW_mutau = decay_width_mutau(ma, couplings, fa, **kwargs_nointegral)
    DW_etau = decay_width_etau(ma, couplings, fa, **kwargs_nointegral)
    DW_charm = decay_width_charm(ma, couplings, fa, **kwargs_nointegral)
    DW_bottom = decay_width_bottom(ma, couplings, fa, **kwargs_nointegral)
    DW_3pis = decay_width_3pi000(ma, couplings, fa, **kwargs)+ decay_width_3pi0pm(ma, couplings, fa, **kwargs)
    DW_3pi000 = decay_width_3pi000(ma, couplings, fa, **kwargs)
    DW_3pi0pm = decay_width_3pi0pm(ma, couplings, fa, **kwargs)
    DW_etapipi = (decay_width_etapipi00(ma, couplings, fa, **kwargs) + decay_width_etapipipm(ma, couplings, fa, **kwargs))
    DW_etapi0pi0 = decay_width_etapipi00(ma, couplings, fa, **kwargs)
    DW_etapippim = decay_width_etapipipm(ma, couplings, fa, **kwargs)
    DW_etappipi = (decay_width_etappipi00(ma, couplings, fa, **kwargs) + decay_width_etappipipm(ma, couplings, fa, **kwargs))
    DW_etappi0pi0 = decay_width_etappipi00(ma, couplings, fa, **kwargs)
    DW_etappippim = decay_width_etappipipm(ma, couplings, fa, **kwargs)
    DW_2w = decay_width_2w(ma, couplings, fa, **kwargs_nointegral)
    DW_gammapipi = decay_width_gammapipi(ma, couplings, fa, **kwargs)
    DW_gluongluon = decay_width_2gluons(ma, couplings, fa, **kwargs_nointegral)
    DW_2photons = decay_width_2gamma(ma, couplings, fa, **kwargs_nointegral)
    DWhadr_nopert = DW_3pis + DW_etapipi + DW_etappipi + DW_gammapipi + DW_2w
    DWhadr_pert = DW_charm + DW_bottom + DW_gluongluon
    if (DWhadr_pert > DWhadr_nopert) and (ma > 1.4):
        DWhadr = DWhadr_pert
        nopert = 0.0
    else:
        DWhadr = DWhadr_nopert
        nopert = 1.0
    DW_sm = DW_elec+DW_muon+DW_tau+DW_2photons+DWhadr+DW_emu+DW_mutau+DW_etau
    if br_dark > 0.0:
        DW_dark = DW_sm/(1-br_dark)*br_dark
    else:
        DW_dark = 0.0
    DWs={
        'e': DW_elec,
        'mu': DW_muon,
        'tau': DW_tau,
        'emu': DW_emu,
        'mutau': DW_mutau,
        'etau': DW_etau,
        'charm': DW_charm * (1.0-nopert),
        'bottom': DW_bottom * (1.0-nopert),
        '3pis': DW_3pis * nopert,
        'pi0pippim': DW_3pi0pm * nopert,
        'pi0pi0pi0': DW_3pi000 * nopert,
        'etapipi': DW_etapipi * nopert,
        'etapi0pi0': DW_etapi0pi0 * nopert,
        'etapippim': DW_etapippim * nopert,
        'etappipi': DW_etappipi * nopert,
        'etappi0pi0': DW_etappi0pi0 * nopert,
        'etappippim': DW_etappippim * nopert,
        'gammapipi': DW_gammapipi * nopert,
        '2omega': DW_2w * nopert,
        'gluongluon': DW_gluongluon * (1.0-nopert),
        '2photons': DW_2photons,
        'hadrons': DWhadr,
        'DW_SM': DW_sm,
        'DW_dark': DW_dark,
        'DW_tot': DW_sm + DW_dark
        }
    return DWs

def total_decay_width(ma, couplings: ALPcouplings, fa, br_dark=0, **kwargs):
    """
    Calculate the total decay width and individual decay widths for various channels.

    Parameters:
    ma (float): Mass of the ALP (Axion-Like Particle).
    couplings (ALPcouplings): Couplings of the ALP to different particles.
    fa (float): Decay constant of the ALP.
    **kwargs: Additional keyword arguments for decay width calculations.

    Returns:
    dict: A dictionary containing the decay widths for various channels:
        - 'e': Decay width to electrons.
        - 'mu': Decay width to muons.
        - 'tau': Decay width to taus.
        - 'charm': Decay width to charm quarks.
        - 'bottom': Decay width to bottom quarks.
        - '3pis': Decay width to three pions.
        - 'etapipi': Decay width to eta and two pions.
        - 'etappipi': Decay width to eta' and two pions.
        - '2omega': Decay width to two omegas.
        - 'gammapipi': Decay width to gamma and two pions.
        - 'gluongluon': Decay width to two gluons.
        - '2photons': Decay width to two photons.
        - 'DW_tot': Total decay width.
    """
    return _total_decay_width(ma, couplings, fa, br_dark, **kwargs)

def BRsalp(ma, couplings: ALPcouplings, fa, br_dark = 0, **kwargs):
    """
    Calculate the branching ratios for various decay channels of the ALP.

    Parameters:
    ma (float): Mass of the ALP (Axion-Like Particle).
    couplings (ALPcouplings): Couplings of the ALP to different particles.
    fa (float): Decay constant of the ALP.
    **kwargs: Additional keyword arguments for decay width calculations.

    Returns:
    dict: A dictionary containing the branching ratios for various channels:
        - 'e': Branching ratio to electrons.
        - 'mu': Branching ratio to muons.
        - 'tau': Branching ratio to taus.
        - 'charm': Branching ratio to charm quarks.
        - 'bottom': Branching ratio to bottom quarks.
        - '3pis': Branching ratio to three pions.
        - 'etapipi': Branching ratio to eta and two pions.
        - 'etappipi': Branching ratio to eta' and two pions.
        - 'gammapipi': Branching ratio to gamma and two pions.
        - '2omega': Branching ratio to two omegas.
        - 'gluongluon': Branching ratio to two gluons.
        - '2photons': Branching ratio to two photons.
        - 'hadrons': Branching ratio to hadrons.
    """
    #kwargs_dw = {k: v for k, v in kwargs.items() if k != 'br_dark'}
    #br_dark = kwargs.get('br_dark', 0.0)
    if br_dark == 1.0:
        return {channel: 0.0 for channel in decay_channels} | {'dark': 1.0}
    DWs = total_decay_width(ma, couplings, fa, br_dark, **kwargs)
    BRs={
        ('electron', 'electron'): DWs['e']/DWs['DW_tot'],
        ('muon', 'muon'): DWs['mu']/DWs['DW_tot'],
        ('tau', 'tau'): DWs['tau']/DWs['DW_tot'],
        ('electron', 'muon'): DWs['emu']/DWs['DW_tot'],
        ('electron', 'tau'): DWs['etau']/DWs['DW_tot'],
        ('muon', 'tau'): DWs['mutau']/DWs['DW_tot'],
        ('charm', 'charm'): DWs['charm']/DWs['DW_tot'],
        ('bottom', 'bottom'): DWs['bottom']/DWs['DW_tot'],
        ('pion', 'pion', 'pion'): DWs['3pis']/DWs['DW_tot'],
        ('pion+', 'pion-', 'pion0'): DWs['pi0pippim']/DWs['DW_tot'],
        ('pion0', 'pion0', 'pion0'): DWs['pi0pi0pi0']/DWs['DW_tot'],
        ('eta', 'pion', 'pion'): DWs['etapipi']/DWs['DW_tot'],
        ('eta', 'pion0', 'pion0'): DWs['etapi0pi0']/DWs['DW_tot'],
        ('eta', 'pion+', 'pion-'): DWs['etapippim']/DWs['DW_tot'],
        ('eta_prime', 'pion', 'pion'): DWs['etappipi']/DWs['DW_tot'],
        ('eta_prime', 'pion0', 'pion0'): DWs['etappi0pi0']/DWs['DW_tot'],
        ('eta_prime', 'pion+', 'pion-'): DWs['etappippim']/DWs['DW_tot'],
        ('photon', 'pion', 'pion'): DWs['gammapipi']/DWs['DW_tot'],
        ('omega', 'omega'): DWs['2omega']/DWs['DW_tot'],
        ('gluon', 'gluon'): DWs['gluongluon']/DWs['DW_tot'],
        ('photon', 'photon'): DWs['2photons']/DWs['DW_tot'],
        ('hadrons',) :DWs['hadrons']/DWs['DW_tot'],
        ('dark',): br_dark
        }
    return BRs
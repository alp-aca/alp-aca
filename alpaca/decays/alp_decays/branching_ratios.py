import numpy as np
from ...rge import ALPcouplings
from .fermion_decays import decay_width_electron, decay_width_muon, decay_width_tau, decay_width_charm, decay_width_bottom
from .hadronic_decays_def import decay_width_3pi000, decay_width_3pi0pm, decay_width_etapipi00, decay_width_etapipipm, decay_width_gammapipi
from .gaugebosons import decay_width_2gamma, decay_width_2gluons
from functools import lru_cache

@lru_cache
def total_decay_width (ma, couplings: ALPcouplings, fa, **kwargs):
    kwargs_nointegral = {k: v for k, v in kwargs.items() if k not in ['nitn_adapt', 'neval_adapt', 'nitn', 'neval', 'cores']}
    DW_elec = decay_width_electron(ma, couplings, fa, **kwargs_nointegral)
    DW_muon = decay_width_muon(ma, couplings, fa, **kwargs_nointegral)
    DW_tau = decay_width_tau(ma, couplings, fa, **kwargs_nointegral)
    DW_charm = decay_width_charm(ma, couplings, fa, **kwargs_nointegral)
    DW_bottom = decay_width_bottom(ma, couplings, fa, **kwargs_nointegral)
    DW_3pis = decay_width_3pi000(ma, couplings, fa, **kwargs)+ decay_width_3pi0pm(ma, couplings, fa, **kwargs)
    DW_etapipi = decay_width_etapipi00(ma, couplings, fa, **kwargs) + decay_width_etapipipm(ma, couplings, fa, **kwargs)
    DW_gammapipi = decay_width_gammapipi(ma, couplings, fa, **kwargs)
    DW_gluongluon = decay_width_2gluons(ma, couplings, fa, **kwargs_nointegral)
    DW_2photons = decay_width_2gamma(ma, couplings, fa, **kwargs_nointegral)
    DWs={'e': DW_elec, 'mu': DW_muon, 'tau': DW_tau, 'charm': DW_charm, 'bottom': DW_bottom, '3pis': DW_3pis, 'etapipi': DW_etapipi, 'gammapipi': DW_gammapipi, 'gluongluon': DW_gluongluon, '2photons': DW_2photons, 'DW_tot': DW_elec+DW_muon+DW_tau+DW_charm+DW_bottom+DW_3pis+DW_etapipi+DW_gammapipi+DW_gluongluon+DW_2photons}
    return DWs

def BRsalp(ma, couplings: ALPcouplings, fa, **kwargs):
    DWs = total_decay_width(ma, couplings, fa, **kwargs)
    BRs={'e': DWs['e']/DWs['DW_tot'], 'mu': DWs['mu']/DWs['DW_tot'], 'tau': DWs['tau']/DWs['DW_tot'], 'charm': DWs['charm']/DWs['DW_tot'], 'bottom': DWs['bottom']/DWs['DW_tot'], '3pis': DWs['3pis']/DWs['DW_tot'], 'etapipi': DWs['etapipi']/DWs['DW_tot'], 'gammapipi': DWs['gammapipi']/DWs['DW_tot'], 'gluongluon': DWs['gluongluon']/DWs['DW_tot'], '2photons': DWs['2photons']/DWs['DW_tot'], 'hadrons':(DWs['3pis'] + DWs['etapipi'] + DWs['gammapipi'])/DWs['DW_tot']}
    return BRs

# def BR_electron(ma, fa, couplings: ALPcouplings):
#     DW_elec = decay_width_electron(ma, couplings, fa)
#     return DW_elec/total_decay_width(ma, fa, couplings)

# def BR_muon(ma, fa, couplings: ALPcouplings):
#     DW_muon = decay_width_muon(ma, couplings, fa)
#     return DW_muon/total_decay_width(ma, fa, couplings)

# def BR_tau(ma, fa, couplings: ALPcouplings):
#     DW_tau = decay_width_tau(ma, couplings, fa)
#     return DW_tau/total_decay_width(ma, fa, couplings)

# def BR_charm(ma, fa, couplings: ALPcouplings):
#     DW_charm = decay_width_charm(ma, couplings, fa)
#     return DW_charm/total_decay_width(ma, fa, couplings)

# def BR_bottom(ma, fa, couplings: ALPcouplings):
#     DW_bottom = decay_width_bottom(ma, couplings, fa)
#     return DW_bottom/total_decay_width(ma, fa, couplings)

# def BR_3pis(ma, fa, couplings: ALPcouplings):
#     DW_3pis = decay_width_3pi000(ma, couplings, fa)+ decay_width_3pi0pm(ma, couplings, fa)
#     return DW_3pis/total_decay_width(ma, fa, couplings)

# def BR_etapipi(ma, fa, couplings: ALPcouplings):
#     DW_etapipi = decay_width_etapipi00(ma, couplings, fa) + decay_width_etapipipm(ma, couplings, fa)
#     return DW_etapipi/total_decay_width(ma, fa, couplings)

# def BR_gammapipi(ma, fa, couplings: ALPcouplings):
#     DW_gammapipi = decay_width_gammapipi(ma, couplings, fa)
#     return DW_gammapipi/total_decay_width(ma, fa, couplings)

# def BR_gluongluon(ma, fa, couplings: ALPcouplings):
#     DW_gluongluon = decay_width_2gluons(ma, couplings, fa)
#     return DW_gluongluon/total_decay_width(ma, fa, couplings)

# def BR_2photons(ma, fa, couplings: ALPcouplings):
#     DW_2photons = decay_width_2gamma(ma, couplings, fa)
#     return DW_2photons/total_decay_width(ma, fa, couplings)



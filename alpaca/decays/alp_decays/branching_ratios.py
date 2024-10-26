import numpy as np
from ...rge import ALPcouplings
from .fermion_decays import decay_width_electron, decay_width_muon, decay_width_tau, decay_width_charm, decay_width_bottom
from .hadronic_decays_def import decay_width_3pi000, decay_width_3pi0pm, decay_width_etapipi00, decay_width_etapipipm, decay_width_gammapipi, decay_width_gluongluon
from .gaugebosons import decay_width_2gamma

def total_decay_width (ma, fa, couplings: ALPcouplings):
    DW_elec = decay_width_electron(ma, couplings, fa)
    DW_muon = decay_width_muon(ma, couplings, fa)
    DW_tau = decay_width_tau(ma, couplings, fa)
    DW_charm = decay_width_charm(ma, couplings, fa)
    DW_bottom = decay_width_bottom(ma, couplings, fa)
    DW_3pis = decay_width_3pi000(ma, couplings, fa)+ decay_width_3pi0pm(ma, couplings, fa)
    DW_etapipi = decay_width_etapipi00(ma, couplings, fa) + decay_width_etapipipm(ma, couplings, fa)
    DW_gammapipi = decay_width_gammapipi(ma, couplings, fa)
    DW_gluongluon = decay_width_gluongluon(ma, fa)
    DW_2photons = decay_width_2gamma(ma, couplings, fa)
    return DW_elec + DW_muon + DW_tau + DW_charm + DW_bottom + DW_3pis + DW_etapipi + DW_gammapipi + DW_gluongluon + DW_2photons

def BR_electron(ma, fa, couplings: ALPcouplings):
    DW_elec = decay_width_electron(ma, couplings, fa)
    return DW_elec/total_decay_width(ma, fa, couplings)

def BR_muon(ma, fa, couplings: ALPcouplings):
    DW_muon = decay_width_muon(ma, couplings, fa)
    return DW_muon/total_decay_width(ma, fa, couplings)

def BR_tau(ma, fa, couplings: ALPcouplings):
    DW_tau = decay_width_tau(ma, couplings, fa)
    return DW_tau/total_decay_width(ma, fa, couplings)

def BR_charm(ma, fa, couplings: ALPcouplings):
    DW_charm = decay_width_charm(ma, couplings, fa)
    return DW_charm/total_decay_width(ma, fa, couplings)

def BR_bottom(ma, fa, couplings: ALPcouplings):
    DW_bottom = decay_width_bottom(ma, couplings, fa)
    return DW_bottom/total_decay_width(ma, fa, couplings)

def BR_3pis(ma, fa, couplings: ALPcouplings):
    DW_3pis = decay_width_3pi000(ma, couplings, fa)+ decay_width_3pi0pm(ma, couplings, fa)
    return DW_3pis/total_decay_width(ma, fa, couplings)

def BR_etapipi(ma, fa, couplings: ALPcouplings):
    DW_etapipi = decay_width_etapipi00(ma, couplings, fa) + decay_width_etapipipm(ma, couplings, fa)
    return DW_etapipi/total_decay_width(ma, fa, couplings)

def BR_gammapipi(ma, fa, couplings: ALPcouplings):
    DW_gammapipi = decay_width_gammapipi(ma, couplings, fa)
    return DW_gammapipi/total_decay_width(ma, fa, couplings)

def BR_gluongluon(ma, fa, couplings: ALPcouplings):
    DW_gluongluon = decay_width_gluongluon(ma, fa)
    return DW_gluongluon/total_decay_width(ma, fa, couplings)

def BR_2photons(ma, fa, couplings: ALPcouplings):
    DW_2photons = decay_width_2gamma(ma, couplings, fa)
    return DW_2photons/total_decay_width(ma, fa, couplings)



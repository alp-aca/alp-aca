import numpy as np
from ...rge import ALPcouplings
from .fermion_decays import decay_width_electron, decay_width_muon, decay_width_tau, decay_width_charm, decay_width_bottom
from .hadronic_decays_def import decay_width_3pi000, decay_width_3pi0pm, decay_width_etapipi00, decay_width_etapipipm, decay_width_gammapipi, decay_width_gluongluon

def total_decay_width (ma, fa, couplings: ALPcouplings):
    return 0
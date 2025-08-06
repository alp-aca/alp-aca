#File with all possible theoretical predictions
from ..biblio.biblio import citations
from ..decays.decays import parse
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import flavio
from functools import cache
import numpy as np
from ..constants import hbar_GeVps

#################################### INVISIBLE FINAL STATES ####################################
#B+->K+ nu nu 
    #arXiv: 2207.13371
    #Branching ratio
def theo_BptoKpnunu():
    citations.register_inspire('Parrott:2022zte')
    value = 5.67e-6
    sigmal = 0.38e-6
    sigmar = sigmal
    return value, sigmal, sigmar

#B+->K*+ nu nu 
    #arXiv: 0902.0160
    #Branching ratio
def theo_BptoKpstarnunu():
    citations.register_inspire('Altmannshofer:2009ma')
    value = 6.8e-6
    sigmal = 1.1e-6
    sigmar = 1.0e-6
    return value, sigmal, sigmar

#B0->K*0 nu nu 
    #arXiv: 1409.4557
    #Branching ratio
def theo_B0toK0starnunu():
    citations.register_inspire('Buras:2014fpa')
    value = 9.2e-6
    sigmal = 1.0e-6
    sigmar = 1.0e-6
    return value, sigmal, sigmar

#K+->pi+ nu nu 
    #arXiv: 1503.02693
    #Branching ratio
def theo_Ktopinunu():
    citations.register_inspire('Buras:2015qea')
    value = 9.11e-11
    sigmal = 0.72e-11
    sigmar = sigmal
    return value, sigmal, sigmar

#D0->pi0 nu nu
def theo_D0topi0nunu():
    citations.register_inspire('Burdman:2001tf')
    value = 5.0e-16
    sigmal = 0
    sigmar = sigmal
    return value, sigmal, sigmar



#Bs -> mu mu
    #arXiv: 2407.03810
    #Branching ratio
def theo_Bstomumu():
    citations.register_inspire('Czaja:2024the')
    value = 3.64e-9
    sigmal = 0.12e-9
    sigmar = sigmal
    return value, sigmal, sigmar

@cache
def get_th_uncert(process: str | tuple, N: int = 50) -> float:
    if process == 'delta_mB0':
        return flavio.sm_uncertainty('DeltaM_d', N=N)/hbar_GeVps
    elif process == 'delta_mBs':
        return flavio.sm_uncertainty('DeltaM_s', N=N)/hbar_GeVps
    elif process == 'delta_mK0':
        return 1.8e-3
    elif process == 'epsK':
        return flavio.sm_uncertainty('eps_K', N=N)
    elif process == 'x_D0':
        return flavio.sm_uncertainty('x_D', N=N)
    elif process == 'phi12_D0':
        return 0
    elif process == 'ASL_B0':
        return flavio.sm_uncertainty('a_fs_d', N=N)
    elif process == 'ASL_Bs':
        return flavio.sm_uncertainty('a_fs_s', N=N)
    elif process in ['B+', 'B-', 'B0', 'Bd0', 'D+', 'D-', 'D0', 'Ds+', 'Ds-', 'K+', 'K-', 'KL', 'KS', 'K0L', 'K0S']:
        return 0
    if isinstance(process, str):
        transition = process
    else:
        transition = process[0]
        args = process[1:]
    initial, final = parse(transition)
    if initial == ['Bs'] and final == ['electron', 'electron']:
        return flavio.sm_uncertainty('BR(Bs->ee)', N=N)
    elif initial == ['Bs'] and final == ['muon', 'muon']:
        return theo_Bstomumu()[1]
    elif initial == ['Bs'] and final == ['tau', 'tau']:
        return flavio.sm_uncertainty('BR(Bs->tautau)', N=N)
    elif initial == ['B0'] and final == ['electron', 'electron']:
        return flavio.sm_uncertainty('BR(B0->ee)', N=N)
    elif initial == ['B0'] and final == ['muon', 'muon']:
        return flavio.sm_uncertainty('BR(B0->mumu)', N=N)
    elif initial == ['B0'] and final == ['tau', 'tau']:
        return flavio.sm_uncertainty('BR(B0->tautau)', N=N)
    elif initial == ['KL'] and final == ['electron', 'electron']:
        from ..constants import re_ae, me, mKL, re_ae_error, br_KLgammagamma
        a_em = 1/137
        # Only the contribution of the real part of the long distance amplitude is considered
        # -1.8 is the approximate value of the real part of the short distance amplitude
        return 2*a_em**2/np.pi**2*me**2/mKL**2 *(2*np.abs(re_ae-1.8)*re_ae_error)*br_KLgammagamma*np.sqrt(1-4*me**2/mKL**2)
    elif initial == ['KL'] and final == ['muon', 'muon']:
        from ..constants import re_amu, mmu, mKL, re_amu_error, br_KLgammagamma
        a_em = 1/137
        # Only the contribution of the real part of the long distance amplitude is considered
        # -1.8 is the approximate value of the real part of the short distance amplitude
        return 2*a_em**2/np.pi**2*mmu**2/mKL**2 *(2*np.abs(re_amu-1.8)*re_amu_error)*br_KLgammagamma*np.sqrt(1-4*mmu**2/mKL**2)
    elif initial == ['KS'] and final == ['photon', 'photon']:
        return 0.32e-6
    else:
        return 0.0
    
def get_th_value(transition: str) -> float:
    return 0.0
import numpy as np

from ...rge import ALPcouplings
from ...citations import citations

def Ktopia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mK, mpi_pm
    from ...common import f0_Kpi, kallen
    if ma > mK-mpi_pm:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdV'][0,1]/f_a
    kallen_factor = kallen(1, mpi_pm**2/mK**2, ma**2/mK**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mK**3*abs(gq_eff)**2/(64*np.pi) * f0_Kpi(ma**2)**2*np.sqrt(kallen_factor)*(1-mpi_pm**2/mK**2)**2

def KLtopi0a(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mKL, mpi0
    from ...common import f0_Kpi, kallen
    if ma > mKL-mpi0:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdV'][0,1]/f_a
    kallen_factor = kallen(1, mpi0**2/mKL**2, ma**2/mKL**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mKL**3*np.imag(gq_eff)**2/(64*np.pi) * f0_Kpi(ma**2)**2*np.sqrt(kallen_factor)*(1-mpi0**2/mKL**2)**2

def KStopi0a(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mKS, mpi0
    from ...common import f0_Kpi, kallen
    if ma > mKS-mpi0:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdV'][0,1]/f_a
    kallen_factor = kallen(1, mpi0**2/mKS**2, ma**2/mKS**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mKS**3*np.real(gq_eff)**2/(64*np.pi) * f0_Kpi(ma**2)**2*np.sqrt(kallen_factor)*(1-mpi0**2/mKS**2)**2

def BtoKa(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mK, mB
    from ...common import f0_BK, kallen
    if ma > mB-mK:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdV'][1,2]/f_a
    kallen_factor = kallen(1, mK**2/mB**2, ma**2/mB**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB**3*abs(gq_eff)**2/(64*np.pi) * f0_BK(ma**2)**2*np.sqrt(kallen_factor)*(1-mK**2/mB**2)**2

def B0toKsta(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mKst0, mB0
    from ...common import A0_BKst, kallen
    if ma > mB0-mKst0:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdA'][1,2]/f_a
    kallen_factor = kallen(1, mKst0**2/mB0**2, ma**2/mB0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB0**3*abs(gq_eff)**2/(64*np.pi) * A0_BKst(ma**2)**2 * kallen_factor**1.5

def sigmaNR(ma: float, couplings: ALPcouplings, s: float, f_a: float=1000,**kwargs):
    citations.register_inspire('Merlo:2019anv')
    citations.register_inspire('DiLuzio:2024jip')
    from ...constants import hbarc2_GeV2pb
    from ...common import alpha_em
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gaphoton = coup_low['cgamma']*alpha_em(np.sqrt(s))/(np.pi*f_a)
    return hbarc2_GeV2pb*(((alpha_em(np.sqrt(s))*np.abs(gaphoton)**2)/24)*(1-(ma**2)/s)**3)

def BR_Vagamma(ma: float, couplings: ALPcouplings, mV: float, BeeV: float, quark: str, f_a: float=1000, **kwargs):
    citations.register_inspire('Merlo:2019anv')
    citations.register_inspire('DiLuzio:2024jip')
    citations.register_inspire('Hwang:1997ie') # Eliminate fV in favour of BR(V->ee)
    from ...common import alpha_em
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    if quark == 'b':
        gaff = 0.5*coup_low['cdA'][2,2]/f_a
    elif quark=='c':
        gaff = 0.5*coup_low['cuA'][1,1]/f_a
    else:
        raise ValueError("Q must be -1/3 or 2/3")
    gaphoton = coup_low['cgamma']*alpha_em(mV)/(np.pi*f_a)
    return mV**2/(32*np.pi*alpha_em(mV))*BeeV*(1-(ma**2)/mV**2)*np.abs((gaphoton)*(1-ma**2/mV**2)-2*gaff)**2

def sigmapeak(mV, BeeV):
     from ...constants import hbarc2_GeV2pb
     return (12*np.pi*BeeV/(mV**2))*hbarc2_GeV2pb

def Mixed_QuarkoniaSearches(ma: float, couplings: ALPcouplings, mV: float, quark: str, f_a: float=1000, **kwargs):
    from ...constants import mJpsi, mUpsilon3S, BeeJpsi, BeeUpsilon3S
    if mV == mJpsi:
        Corr_Factor=0.03075942
        BeeV = BeeJpsi
    elif mV == mUpsilon3S:
        Corr_Factor=0.0023112
        BeeV = BeeUpsilon3S
    else:
        raise ValueError("mV must be mJpsi or mUpsilon3S")
    
    sigma_peak=sigmapeak(mV, BeeV)
    return BR_Vagamma(ma, couplings, mV, BeeV, quark, f_a, **kwargs)+sigmaNR(ma, couplings, mV**2, f_a, **kwargs)/(Corr_Factor*sigma_peak)
        

import numpy as np

from ...common import f0_BK, f0_Kpi, A0_BKst, alpha_em, mJPSI,muppsilon3s, BeeJPSI, BeeUppsilon3s, kallen
from ...rge import ALPcouplings
from ...citations import citations

def Ktopia(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    citations.register_inspire('Izaguirre:2016dfi')
    from ...constants import mK, mpi_pm
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gq_eff = coup_low['kD'][0,1]/f_a
    return mK**3*abs(gq_eff)**2/(64*np.pi) * f0_Kpi(ma**2)**2*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))*(1-mpi_pm**2/mK**2)**2

def BtoKa(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    citations.register_inspire('Izaguirre:2016dfi')
    from ...constants import mK, mB
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gq_eff = coup_low['kD'][1,2]/f_a
    return mB**3*abs(gq_eff)**2/(64*np.pi) * f0_BK(ma**2)**2*np.sqrt(kallen(1, mK**2/mB**2, ma**2/mB**2))*(1-mK**2/mB**2)**2

def B0toKsta(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    citations.register_inspire('Izaguirre:2016dfi')
    from ...constants import mKst0, mB0
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gq_eff = coup_low['kD'][1,2]/f_a
    return mB0**3*abs(gq_eff)**2/(64*np.pi) * A0_BKst(ma**2)**2 * kallen(1, mKst0**2/mB0**2, ma**2/mB0**2)**1.5

def sigmaNR(ma: float, couplings: ALPcouplings, s: float, f_a: float=1000,**kwargs):
    citations.register_inspire('Merlo:2019anv')
    citations.register_inspire('DiLuzio:2024jip')
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    gaphoton = coup_low['cgamma']*alpha_em(np.sqrt(s))/(np.pi*f_a)
    return (((alpha_em(np.sqrt(s))*gaphoton**2)/24)*(1-(ma**2)/s)**3)*0.389379e9

def BR_Vagamma(ma: float, couplings: ALPcouplings, s: float, mV: float, fV: float, Q: float, GammaV: float, f_a: float=1000, **kwargs):
    citations.register_inspire('Merlo:2019anv')
    citations.register_inspire('DiLuzio:2024jip')
    coup_low = couplings.match_run(ma, 'kF_below', **kwargs)
    if Q == -1/3:
        gaff = coup_low['kD'][2,2]/f_a
    elif Q==2/3:
        gaff = coup_low['kU'][1,1]/f_a
    else:
        raise ValueError("Q must be -1/3 or 2/3")
    gaphoton = coup_low['cgamma']*alpha_em(np.sqrt(mV))/(np.pi*f_a)
    return ((alpha_em(mV)*(Q**2)*mV*fV**2)/(24*GammaV))*(1-(ma**2)/mV**2)*((gaphoton)*(1-ma**2/mV**2)-2*gaff)**2

def sigmapeak(mV, BeeV):
     return (12*np.pi*BeeV/(mV**2))*0.389379*10**9

def Mixed_QuarkoniaSearches(ma: float, couplings: ALPcouplings, s: float, mV: float, fV: float, Q: float, GammaV: float, f_a: float=1000, **kwargs):
    if mV == mJPSI:
        Corr_Factor=0.03075942
        sigma_peak=sigmapeak(mV, BeeJPSI)
    elif mV == muppsilon3s:
        Corr_Factor=0.0023112
        sigma_peak=sigmapeak(mV, BeeUppsilon3s)
    else:
        raise ValueError("mV must be mJPSI or muppsilon3s")
    
    return BR_Vagamma(ma, couplings, s, mV, fV, Q, GammaV, f_a, **kwargs)+sigmaNR(ma, couplings, s, f_a, **kwargs)/(Corr_Factor*sigma_peak)
        

import numpy as np

from ...rge import ALPcouplings
from ...rge.runSM import runSM
from ...citations import citations

def Kminustopia(ma: float, couplings: ALPcouplings, f_a: float=1000, delta8=0, **kwargs):
    from ...constants import mK, mpi_pm, g8, fpi, GammaK, GF
    from ... common import kallen
    if ma > mK-mpi_pm:
        return 0
    citations.register_inspire('Bauer:2021wjo')
    coupl_low = couplings.match_run(ma, 'kF_below', **kwargs)
    cg = coupl_low['cg']
    cuu = coupl_low['ku'][0,0]-coupl_low['kU'][0,0]
    cdd = coupl_low['kd'][0,0]-coupl_low['kD'][0,0]
    css = coupl_low['kd'][1,1]-coupl_low['kD'][1,1]
    kd = coupl_low['kd'][0,0]
    kD = coupl_low['kD'][0,0]
    ks = coupl_low['kd'][1,1]
    kS = coupl_low['kD'][1,1]
    parsSM = runSM(ma)
    Vckm = parsSM['CKM']
    N8 = -g8*GF/np.sqrt(2)*np.conj(Vckm[0,0])*Vckm[0,1]*fpi**2*(np.cos(delta8)+1j*np.sin(delta8))

    chiral_contrib = 16*cg*(mK**2-mpi_pm**2)*(mK**2-ma**2)/(4*mK**2-mpi_pm**2-3*ma**2)
    chiral_contrib += 6*(cuu+cdd-2*css)*ma**2*(mK**2-ma**2)/(4*mK**2-mpi_pm**2-3*ma**2)
    chiral_contrib += (2*cuu+cdd+css)*(mK**2-mpi_pm**2-ma**2) + 4*css*ma**2
    chiral_contrib += (kd+kD-ks-kS)*(mK**2+mpi_pm**2-ma**2)

    amp = N8*chiral_contrib/(4*f_a)-(mK**2-mpi_pm**2)/(2*f_a)*(coupl_low['kd'][0,1]+coupl_low['kD'][0,1])
    return np.abs(amp)**2/(16*np.pi*mK)*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))/GammaK

def Kplustopia(ma: float, couplings: ALPcouplings, f_a: float=1000, delta8=0, **kwargs):
    from ...constants import mK, mpi_pm, g8, fpi, GammaK, GF
    from ... common import kallen
    if ma > mK-mpi_pm:
        return 0
    citations.register_inspire('Bauer:2021wjo')
    coupl_low = couplings.match_run(ma, 'kF_below', **kwargs)
    cg = coupl_low['cg']
    cuu = coupl_low['ku'][0,0]-coupl_low['kU'][0,0]
    cdd = coupl_low['kd'][0,0]-coupl_low['kD'][0,0]
    css = coupl_low['kd'][1,1]-coupl_low['kD'][1,1]
    kd = coupl_low['kd'][0,0]
    kD = coupl_low['kD'][0,0]
    ks = coupl_low['kd'][1,1]
    kS = coupl_low['kD'][1,1]
    parsSM = runSM(ma)
    Vckm = parsSM['CKM']
    N8 = -g8*GF/np.sqrt(2)*np.conj(Vckm[0,0])*Vckm[0,1]*fpi**2*(np.cos(delta8)+1j*np.sin(delta8))

    chiral_contrib = 16*cg*(mK**2-mpi_pm**2)*(mK**2-ma**2)/(4*mK**2-mpi_pm**2-3*ma**2)
    chiral_contrib += 6*(cuu+cdd-2*css)*ma**2*(mK**2-ma**2)/(4*mK**2-mpi_pm**2-3*ma**2)
    chiral_contrib += (2*cuu+cdd+css)*(mK**2-mpi_pm**2-ma**2) + 4*css*ma**2
    chiral_contrib += (kd+kD-ks-kS)*(mK**2+mpi_pm**2-ma**2)

    amp = N8*chiral_contrib/(4*f_a)-(mK**2-mpi_pm**2)/(2*f_a)*(coupl_low['kd'][1,0]+coupl_low['kD'][1,0])
    return np.abs(amp)**2/(16*np.pi*mK)*np.sqrt(kallen(1, mpi_pm**2/mK**2, ma**2/mK**2))/GammaK

def KLtopia(ma: float, couplings: ALPcouplings, f_a: float=1000, delta8=0, **kwargs):
    from ...constants import mKL, mpi0, g8, fpi, epsilonKaon, phiepsilonKaon, GammaKL, GF
    from ... common import kallen
    if ma > mKL-mpi0:
        return 0
    citations.register_inspire('Bauer:2021mvw')
    coupl_low = couplings.match_run(ma, 'kF_below', **kwargs)
    cg = coupl_low['cg']
    cuu = coupl_low['ku'][0,0]-coupl_low['kU'][0,0]
    cdd = coupl_low['kd'][0,0]-coupl_low['kD'][0,0]
    css = coupl_low['kd'][1,1]-coupl_low['kD'][1,1]
    kd = coupl_low['kd'][0,0]
    kD = coupl_low['kD'][0,0]
    ks = coupl_low['kd'][1,1]
    kS = coupl_low['kD'][1,1]
    parsSM = runSM(ma)
    Vckm = parsSM['CKM']
    N8 = -g8*GF/np.sqrt(2)*np.conj(Vckm[0,0])*Vckm[0,1]*fpi**2*(np.cos(delta8)+1j*np.sin(delta8))
    eps = epsilonKaon*(np.cos(phiepsilonKaon)+1j*np.sin(phiepsilonKaon))

    chiral_contrib = 16*cg*(mKL**2-mpi0**2)*(mKL**2-ma**2)/(4*mKL**2-mpi0**2-3*ma**2)
    chiral_contrib -= 2*(cuu+cdd-2*css)*ma**2*(mKL**2-ma**2)/(4*mKL**2-mpi0**2-3*ma**2)
    chiral_contrib += (3*cdd+css)*(mKL**2-mpi0**2)+(2*cuu-cdd-css)*ma**2
    chiral_contrib -= 2*(cuu-cdd)*ma**2*(mKL**2-ma**2)/(mpi0**2-ma**2)
    chiral_contrib += (kd+kD-ks-kS)*(mKL**2+mpi0**2-ma**2)

    amp_K0bar = N8*chiral_contrib/(4*f_a*2**0.5)-(mKL**2-mpi0**2)/(2*f_a*2**0.5)*(coupl_low['kd'][0,1]+coupl_low['kD'][0,1])
    amp_K0 = -N8*chiral_contrib/(4*f_a*2**0.5)+(mKL**2-mpi0**2)/(2*f_a*2**0.5)*(coupl_low['kd'][1,0]+coupl_low['kD'][1,0])
    amp = ((1+eps)*amp_K0+(1-eps)*amp_K0bar)/np.sqrt(2*(1+np.abs(eps)**2))
    return np.abs(amp)**2/(16*np.pi*mKL)*np.sqrt(kallen(1, mpi0**2/mKL**2, ma**2/mKL**2))/GammaKL

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

def B0toKa(ma: float, couplings: ALPcouplings, f_a: float=1000, **kwargs):
    from ...constants import mK0, mB0
    from ...common import f0_BK, kallen
    if ma > mB0-mK0:
        return 0
    citations.register_inspire('Izaguirre:2016dfi')
    coup_low = couplings.match_run(ma, 'VA_below', **kwargs)
    gq_eff = coup_low['cdV'][1,2]/f_a
    kallen_factor = kallen(1, mK0**2/mB0**2, ma**2/mB0**2)
    kallen_factor = np.where(kallen_factor>0, kallen_factor, np.nan)
    return mB0**3*abs(gq_eff)**2/(64*np.pi) * f0_BK(ma**2)**2*np.sqrt(kallen_factor)*(1-mK0**2/mB0**2)**2

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
        

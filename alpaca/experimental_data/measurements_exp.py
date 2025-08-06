#File with all possible experimental measurement
import os
import numpy as np
from scipy.stats import chi2
import particle.literals as particles
from ..biblio.biblio import citations
from ..constants import mUpsilon3S
from .classes import MeasurementBase, MeasurementConstantBound, MeasurementInterpolatedBound, MeasurementInterpolated, MeasurementDisplacedVertexBound, MeasurementBinned, rmax_belle, rmax_besIII, MeasurementConstant
from ..decays.particles import particle_aliases
from ..decays.decays import parse
from ..constants import mB, mB0, mBs, mK, mtau, mKst0, mKL, mpi0, mpi_pm, mphi, mZ, mDplus, mDs, mrho, mD0, meta, mUpsilon1S, mJpsi, me, mmu
# Get the directory of the current script
current_dir = os.path.dirname(__file__)

#Meson mass (useful)
mKl = 0.493 #GeV
mD0 = 1.864 #GeV

######### Auxiliary functions #########
#Document reading
def data_reading(filename_path):
    q2min = []
    q2max = []
    value = []
    # Open the file in read mode
    with open(filename_path, 'r') as file:
        # Read the file line by line
        for line in file:
            aux = line.strip().split('\t')
            q2min.append(float(aux[0]))
            q2max.append(float(aux[1]))
            value.append(float(aux[2]))
    return q2min, q2max, value

#Bin selection
def bin_selection(x, qmin, qmax, value, sigmal, sigmar):
    values = 0
    sigmals = 0
    sigmars = 0 
    for ii in range(len(qmin)):
        if x > qmin[ii] and x < qmax[ii]:
            values = value[ii]
            sigmals = sigmal[ii]
            sigmars = sigmar[ii]
            break
    return values, sigmals, sigmars

#Confidence level and sigma calculation
def sigma(cl, df, param):
    #INPUT:
        #cl: Confidence level of measurement
        #df: Degrees of freedom (df)
        #param: Measured quantity
    #OUTPUT:
        #Value of standard deviation of the measurement with said confidence level
    p_value = 1 - cl
    # Calculate the chi-squared value
    chi_squared_value = chi2.ppf(1 - p_value, df)
    return param/np.sqrt(chi_squared_value)

#################################### INVISIBLE SEARCHES ####################################
invisible = "invisible/"
#BELLEII B+->K+ nu nu 2023
    #Experiment: BELLE II
    #arXiv: 2311.14647
    #Branching ratio
def belleII_BtoKnunu(x):
    #INPUT:
        #x:
    citations.register_inspire('Belle-II:2023esi') 
    q2min = [0]
    q2max = [mB0]
    value = [2.3e-5]
    sigmal = [0.7e-5]
    sigmar = [0.7e-5]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

#NA62 K+->pi+ nu nu 2021
    #Experiment: NA62
    #arXiv: 2103.15389
    #Branching ratio
# def na62_Ktopinunu(x):
#     citations.register_inspire('NA62:2021zjw') 
#     q2min = [0]
#     q2max = [mK]
#     value = [10.6e-11]
#     sigmal = [4.1e-11]
#     sigmar = [3.5e-11]
#     values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
#     return values, sigmals, sigmars

# #NA62 K+->pi+ pi0(->X) 2020 
#     #Experiment: NA62
#     #arXiv: 2010.07644
#     #Branching ratio
# def na62_pi0toinv(x):
#     citations.register_inspire('NA62:2020pwi') 
#     q2min = [0.110]
#     q2max = [0.155]
#     param = [4.4e-9]
#     sigmap = sigma(0.9, 1, param)
#     value = [0] #Estimated value
#     values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
#     return values, sigmals, sigmars

na62_Ktopiinv22 = MeasurementInterpolatedBound(
    ['Guadagnoli:2025xnt', 'NA62:2024pjp'],
    os.path.join(current_dir, invisible, 'Kpialp_na62.txt'),
    'invisible',
    rmax = 14000,
    mass_parent=mK,
    mass_sibling=mpi_pm,
    lab_boost=75*mK
)

na62_Ktopiinv = MeasurementDisplacedVertexBound(
    'NA62:2020pwi',
    os.path.join(current_dir, invisible, 'na62_kpiInv.npy'),
    decay_type = 'invisible'
)

na62_Ktopiinv_2025 = MeasurementDisplacedVertexBound(
    'NA62:2025upx',
    os.path.join(current_dir, invisible, 'na62_kpiInv_2025.npy'),
    decay_type = 'invisible'
)

na62_Ktopipi0 = MeasurementDisplacedVertexBound(
    'NA62:2020pwi',
    os.path.join(current_dir, invisible, 'na62_kpipi0.npy'),
    decay_type = 'invisible'
)

e949_Ktopiinv = MeasurementDisplacedVertexBound(
    'BNL-E949:2009dza',
    os.path.join(current_dir, invisible, 'e949_kpiInv.npy'),
    decay_type = 'invisible'
)

#J-PARC KOTO KL->pi0 nu nu
    #Experiment: KOTO
    #arXiv: 1810.09655
    #Branching ratio

# def koto_kltopi0nunu(x):
#     citations.register_inspire('KOTO:2018dsc') 
#     q2min = [0]
#     q2max = [0.261]
#     param = [3.0e-9]
#     sigmap = sigma(0.9, 1, param)
#     value = [0] #Estimated value
#     values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
#     return values, sigmals, sigmars
koto_kltopi0inv = MeasurementInterpolatedBound(
    'KOTO:2018dsc',
    os.path.join(current_dir, invisible, 'koto_KLpiInv.txt'),
    'invisible',
    conf_level=0.9,
    rmax=414.8,
    lab_boost=1.5/mK,
    mass_parent=mKL,
    mass_sibling=mpi0
)

#J-PARC KOTO KL->pi0 inv
    #Experiment: KOTO
    #arXiv: 1810.09655
    #Branching ratio
# def koto_kltopi0inv(x):
#     citations.register_inspire('KOTO:2018dsc') 
#     q2min = [0]
#     q2max = [0.261]
#     param = [2.4e-9]
#     sigmap = sigma(0.9, 1, param)
#     value = [0] #Estimated value
#     values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
#     return values, sigmals, sigmars

#BaBar B+->K+ nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_bptokpnunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [3.7e-5]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars

#BaBar B+->K*+ nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_bptokstarpnunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [11.6e-5]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars

#BaBar B0->K0 nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_b0tok0nunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [8.1e-5]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars

#BaBar B0->K*0 nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_b0tokstar0nunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [9.3e-5]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars

belleII_bptoknunu_lightmediator = MeasurementInterpolated(['Altmannshofer:2023hkn', 'Belle-II:2023esi'], os.path.join(current_dir, invisible, 'BelleII_BtoK_bestfit.txt'), 'invisible', rmax=100, lab_boost=0.28, mass_parent=mB, mass_sibling=mK)

babar_btoknunu_lightmediator = MeasurementInterpolated(['Altmannshofer:2023hkn', 'BaBar:2013npw'], os.path.join(current_dir, invisible, 'Babar_BtoK_bestfit.txt'), 'invisible', rmax=50, lab_boost=0.469/(1-0.469**2)**0.5, mass_parent=mB, mass_sibling=mK)

# combined_btoknunu_lightmediator = MeasurementInterpolated(['Altmannshofer:2023hkn', 'BaBar:2013npw', 'Belle-II:2023esi'], os.path.join(current_dir, invisible, 'BKinv_combined.txt'), 'invisible', rmax=50, lab_boost=0.469/(1-0.469**2)**0.5, mass_parent=mB, mass_sibling=mK)

babar_btokstarnunu_lightmediator = MeasurementInterpolated(['Altmannshofer:2023hkn', 'BaBar:2013npw'], os.path.join(current_dir, invisible, 'Babar_BKstarinv.txt'), 'invisible', rmax=50, lab_boost=0.469/(1-0.469**2)**0.5, mass_parent=mB, mass_sibling=mKst0)

#BaBar B->K nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_btoknunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [3.2e-5]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars

#BaBar B->K* nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_btokstarnunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [7.9e-5]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars

#BaBar J/psi-> nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_jpsitonunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [3.9e-3]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars

#BaBar psi(2S)-> nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_psi2stonunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    param = [15.5e-3]
    sigmap = sigma(0.9, 1, param)
    value = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmap, sigmap)
    return values, sigmals, sigmars


############ Quarkonia decays ############

#BaBar Upsilon(3S)
    #Experiment: BaBar
    #arXiv:0808.0017
    #Branching ratio
babar_Y3S_inv = MeasurementInterpolatedBound(
    'BaBar:2008aby',
    os.path.join(current_dir, invisible, 'Babar_BR_Y3S_binned.txt'),
    'invisible',
    rmax=50,
    lab_boost=0.469/(1-0.469**2)**0.5, #gamma = 0.469, correspinding to E_electron = 8.6GeV and E_positron = 3.1GeV
    mass_parent=mUpsilon3S,
    mass_sibling=0
    )

#Belle Upsilon(1S)
    #Experiment: Belle
    #arXiv:1809.05222
    #Branching ratio
belle_Y1S_inv = MeasurementInterpolatedBound(
    'Belle:2018pzt',
    os.path.join(current_dir, invisible, 'Belle_BR_Y1S_binned.txt'),
    'invisible',
    rmax=rmax_belle,
    lab_boost=0.42,
    mass_parent=mUpsilon1S,
    mass_sibling=0
)

#BesIII J/psi invisible
    #Experiment: BESIII
    #arXiv:2003.05594
    #Branching ratio

besIII_Jpsiinv = MeasurementInterpolatedBound(
    'BESIII:2020sdo',
    os.path.join(current_dir, invisible, 'BESIII_BR_Jpsi_inv_binned.txt'),
    'invisible',
    rmax = rmax_besIII,
    mass_parent=mJpsi,
    mass_sibling=0,
    lab_boost=11e-3 #Symmetric beams crossing at angle theta=11mrad. pJ/psi = sqrt(s)*sin(theta) approx sqrt(s)*theta
)

#Belle B->h nu nu 2017
    #Experiment: Belle
    #arXiv: 1702.03224
    #Results at 90% confidence level
    #Branching ratio
belle_BchargedtoKchargednunu = MeasurementConstantBound(
    inspire_id='Belle:2017oht',
    decay_type='invisible',
    bound=4e-5,
    conf_level=0.9,
    mass_parent=mB,
    rmax=rmax_belle
)

belle_Bchargedtorhochargednunu = MeasurementConstantBound(
    inspire_id='Belle:2017oht',
    decay_type='invisible',
    bound=3e-5,
    conf_level=0.9,
    mass_parent=mB,
    rmax=rmax_belle,
    lab_boost=0.28
)

belle_Bchargedtopichargednunu = MeasurementConstantBound(
    inspire_id='Belle:2017oht',
    decay_type='invisible',
    bound=1.4e-5,
    conf_level=0.9,
    mass_parent=mB,
    rmax=rmax_belle
)

belle_B0toK0nunu = MeasurementConstantBound(
    inspire_id='Belle:2017oht',
    decay_type='invisible',
    bound=2.6e-5,
    conf_level=0.9,
    mass_parent=mB0,
    rmax=rmax_belle,
    lab_boost=0.28
)

belle_B0toK0starnunu = MeasurementConstantBound(
    inspire_id='Belle:2017oht',
    decay_type='invisible',
    bound=1.8e-5,
    conf_level=0.9,
    mass_parent=mB0,
    rmax=rmax_belle
)

belle_B0topi0nunu = MeasurementConstantBound(
    inspire_id='Belle:2017oht',
    decay_type='invisible',
    bound=9e-6,
    conf_level=0.9,
    mass_parent=mB0,
    rmax=rmax_belle,
    lab_boost=0.28
)

belle_B0torho0nunu = MeasurementConstantBound(
    inspire_id='Belle:2017oht',
    decay_type='invisible',
    bound=4e-5,
    conf_level=0.9,
    mass_parent=mB0,
    rmax=rmax_belle,
    lab_boost=0.28
)

delphi_Bstophinunu = MeasurementConstantBound(
    'DELPHI:1996ohp',
    'invisible',
    5.4e-3,
    mass_parent=mBs,
    mass_sibling=mphi,
    lab_boost=np.sqrt(0.25*mZ**2/mBs**2-1), # Z -> Bs Bs decay
    rmax=24
)

#BESIII D0->pi0 nu nu 2021 
    #Experiment: BESIII
    #arXiv: 2112.14236
    #@ 90% confidence level
    #Branching ratio
besIII_D0topi0nunu = MeasurementConstantBound(
    inspire_id='BESIII:2021slf',
    decay_type='invisible',
    bound=2.1e-4,
    conf_level=0.9,
    mass_parent=mD0,
    mass_sibling=mpi0,
    rmax=rmax_besIII
)

#BelleII e+e- -> gamma a
    #Experiment: BelleII
    #arXiv: 2007.13071
    #@95% confidence level
    #Cross section (pb)

#################################### VISIBLE SEARCHES ####################################

###### Decays to gamma gamma ######

#NA62  K+->pi+ gamma gamma
    #Experiment: NA62
    #arXiv: 1402.4334
# def na62_Ktopigammagamma(x):
#     citations.register_inspire('NA62:2014ybm')
#     q2min = [0.220] #Digamma momentum
#     q2max = [0.354] #Digamma momentum
#     value = [9.65e-7]
#     sigmal = [0.63e-7]
#     sigmar = sigmal
#     values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
#     return values, sigmals, sigmars

#E949  K+->pi+ gamma gamma
    #Experiment: E949
    #arXiv: hep-ex/0505069
def E949_Ktopigammagamma(x):
    citations.register_inspire('E949:2005qiy')
    q2min = [0] #Digamma momentum
    q2max = [0.108] #Digamma momentum
    value = [8.3e-9]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#E787 K+->pi+ gamma gamma
    #Experiment: E787
    #arXiv: hep-ex/9708011
#def E787_Ktopigammagamma(x):
#    citations.register_inspire('E787:1997abk')
#    q2min = [0.196] #Digamma momentum
#    q2max = [0.306] #Digamma momentum
#    value = [6.0e-7]
#    sigmal = [np.sqrt((1.5)**2+(0.7)**2)*1e-7]
#    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmal)
#    return values, sigmals, sigmars

#NA48  KL->pi0 gamma gamma
    #Experiment: NA48
    #arXiv: hep-ex/0205010
def na48_Kltopi0gammagamma(x):
    citations.register_inspire('NA48:2002xke')
    q2min = [0.030] #Digamma momentum
    q2max = [0.110] #Digamma momentum
    value = [0.6e-8]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#KTeV  KL->pi0 gamma gamma
    #Experiment: KTeV
    #arXiv: 0805.0031
def ktev_Kltopi0gammagamma(x):
    citations.register_inspire('KTeV:2008nqz')
    q2min = [0,0.160] #Digamma momentum
    q2max = [0.100,0.363] #Digamma momentum
    value = [1.29e-6]
    sigmas = [np.sqrt(0.03**2+0.05**2)*1e-6]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmas, sigmas)
    return values, sigmals, sigmars



###### Decays to e e (final state) ######

#Brookhaven  K+->pi+ a (-> e+ e-) 
    #Experiment: Brookhaven
    #arXiv: DOI: 10.1103/PhysRevLett.59.2832
def brookhaven_Kptopipee(x):
    citations.register_inspire('Baker:1987gp')
    q2min = [0] #ALP mass
    q2max = [0.100] #
    value = [8e-7]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#KTeV  KL->pi0 e e
    #Experiment: KTeV
    #arXiv: hep-ex/0309072
def ktev_Kltopi0ee(x):
    citations.register_inspire('KTeV:2003sls')
    q2min = [0.140] #Dielectron momentum
    q2max = [0.362] #Dielectron momentum
    value = [2.8e-10]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#NA48/2 K-> pi e e
    #Experiment: NA48/2
    #arXiv: 0903.3130
    #Branching ratio
def na48_Ktopiee(x):
    citations.register_inspire('NA482:2009pfe')
    q2min = [0]
    q2max = [0.354]
    value = [3.11e-7]
    sigmal = [0.12e-7]
    sigmar = [0.12e-7]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

#Belle  B+->pi+ e e
    #Experiment: Belle
    #arXiv: 0804.3656
def belle_Bptopipee(x):
    citations.register_inspire('Belle:2008tjs')
    q2min = [0.140] #Dielectron momentum
    q2max = [5.140] #Dielectron momentum
    value = [8.0e-8]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#LHCb B+->K+ e e
    #Experiment: LHCb
    #arXiv: 2212.09153
def LHCb_BptoKpee_dif(x):
    citations.register_inspire('LHCb:2022vje')
    q2min = [1.1] #q^2
    q2max = [6.0] #q^2
    value = [25.5e-9]
    sigmal = [np.sqrt((1.3)**2+(1.1)**2)*1e-9]
    values, sigmals, sigmars = bin_selection(x**2, q2min, q2max, value, sigmal, sigmal)
    return values, sigmals, sigmars

#LHCb B0->K0* e e
    #Experiment: LHCb
    #arXiv: 2212.09153
def LHCb_B0toK0staree_dif(x):
    citations.register_inspire('LHCb:2022vje')
    q2min = [1.1] #q^2
    q2max = [6.0] #q^2
    value = [33.3e-9]
    sigmal = [np.sqrt((2.7)**2+(2.2)**2)*1e-9]
    values, sigmals, sigmars = bin_selection(x**2, q2min, q2max, value, sigmal, sigmal)
    return values, sigmals, sigmars


#LHCb R(K)
    #Experiment: LHCb
    #arXiv: 2212.09153
def LHCb_Rk(x):
    citations.register_inspire('LHCb:2022vje')
    q2min = [0.1, 1.1] #q^2
    q2max = [1.1, 6.0] #q^2
    value = [0.994, 0.949]
    sigmal = [np.sqrt((0.090)**2+(0.029)**2), np.sqrt((0.042)**2+(0.022)**2)]
    sigmar = [np.sqrt((0.082)**2+(0.027)**2), np.sqrt((0.041)**2+(0.022)**2)]
    values, sigmals, sigmars = bin_selection(x**2, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

#LHCb R(K*)
    #Experiment: LHCb
    #arXiv: 2212.09153
def LHCb_Rkstar(x):
    citations.register_inspire('LHCb:2022vje')
    q2min = [0.1,1.1] #q^2
    q2max = [1.1, 6.0] #q^2
    value = [0.927, 1.027]
    sigmal = [np.sqrt((0.093)**2+(0.036)**2), np.sqrt((0.072)**2+(0.027)**2)]
    sigmar = [np.sqrt((0.087)**2+(0.035)**2), np.sqrt((0.068)**2+(0.026)**2)]
    values, sigmals, sigmars = bin_selection(x**2, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars


lhcb_Dptopipee = MeasurementConstantBound(
    'LHCb:2020car',
    'prompt',
    1600e-9,
    rmin = 150e-4,
    mass_parent=mDplus,
    mass_sibling=mpi0
)

lhcb_DstoKpee = MeasurementConstantBound(
    'LHCb:2020car',
    'prompt',
    850e-9,
    rmin = 150e-4,
    mass_parent=mDs,
    mass_sibling=mK
)

besIII_D0topi0ee = MeasurementConstantBound(
    inspire_id='BESIII:2018hqu',
    decay_type='prompt',
    bound=0.4e-5,
    conf_level=0.9,
    mass_parent=mD0,
    mass_sibling=mpi0,
    rmin=rmax_besIII
)

besIII_D0toetaee = MeasurementConstantBound(
    inspire_id='BESIII:2018hqu',
    decay_type='prompt',
    bound=0.3e-5,
    conf_level=0.9,
    mass_parent=mD0,
    mass_sibling=meta,
    rmin=rmax_besIII
)

e791_D0torho0ee = MeasurementConstantBound(
    'E791:2000jkj',
    'prompt',
    12.4e-5,
    mass_parent=mD0,
    mass_sibling=mrho,
    rmin=1.5
)

####### Decay to mu mu ######
#KTeV KL->pi0 mu mu
    #Experiment: KTeV
    #arXiv: hep-ex/0001006
def ktev_KLtopi0mumu(x):
    citations.register_inspire('KTEV:2000ngj')
    q2min = [0.210] #Dimuon mass
    q2max = [0.350] #Dimuon mass
    value = [3.8e-10]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#CMS B+- -> K+- mu mu
    #Experiment: CMS
    #arXiv:2401.07090
    #Branching ratio
def cms_BchargedtoKchargedmumu(x):
    citations.register_inspire('CMS:2024syx')
    q2min = [0.1, 1.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 11.0, 11.8, 14.82, 16, 17, 18, 19.24]
    q2max = [0.98, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 11.8, 12.5, 16.0, 17.0, 18.0, 19.24, 22.9]
    value = (1e-8)*np.array([2.91, 1.93, 3.06, 2.54, 2.47, 2.53, 2.50, 2.34, 1.62, 1.26, 1.83, 1.57, 2.11, 1.74, 2.02])
    sigmal = (1e-8)*np.array([0.24, 0.20, 0.25, 0.23, 0.24, 0.27, 0.23, 0.25, 0.18, 0.14, 0.17, 0.15, 0.16, 0.15, 0.30])
    sigmar = sigmal
    values, sigmals, sigmars = bin_selection(x**2, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

cms_Bstomumu = MeasurementConstant(
    'CMS:2022mgd',
    'flat',
    3.83e-9,
    np.sqrt(0.36**2+0.21**2)*1e-9,
    np.sqrt(0.38**2+0.24**2)*1e-9,
    max_ma=np.inf
)

cms_B0tomumu = MeasurementConstantBound(
    'CMS:2022mgd',
    'flat',
    1.9e-10,
    max_ma=np.inf,
    conf_level=0.95
)

lhcb_Bstomumu = MeasurementConstant(
    ['LHCb:2021awg', 'LHCb:2021vsc'],
    'flat',
    3.09e-9,
    np.sqrt(0.43**2+0.11**2)*1e-9,
    np.sqrt(0.46**2+0.15**2)*1e-9,
    max_ma=np.inf
)

lhcb_B0tomumu = MeasurementConstantBound(
    ['LHCb:2021awg', 'LHCb:2021vsc'],
    'flat',
    2.6e-10,
    max_ma=np.inf,
    conf_level=0.95
)

atlasHL_Bstomumu = MeasurementConstant(
    'ATLAS:2025eaw',
    'flat',
    3.68e-9,
    0.4e-9,
    0.4e-9,
    max_ma=np.inf
)

atlasHL_B0tomumu = MeasurementConstant(
    'ATLAS:2025eaw',
    'flat',
    1.06e-10,
    0.48e-10,
    0.48e-10,
    max_ma=np.inf
)

cmsHL_Bstomumu = MeasurementConstant(
    'ATLAS:2025lrr',
    'flat',
    3.68e-9,
    0.22e-9,
    0.22e-9,
    max_ma=np.inf,
    bibtex = {'Collaboration:2928094': '''@techreport{Collaboration:2928094,
      author        = "CMS Collaboration",
      collaboration = "CMS",
      title         = "{CMS flavor physics projections for the update of the
                       European Strategy for Particle Physics}",
      institution   = "CERN",
      reportNumber  = "CMS-NOTE-2025-004, CERN-CMS-NOTE-2025-004",
      address       = "Geneva",
      year          = "2025",
      url           = "https://cds.cern.ch/record/2928094",
}'''}
)

cmsHL_B0tomumu = MeasurementConstant(
    'ATLAS:2025lrr',
    'flat',
    1.06e-10,
    0.12e-10,
    0.12e-10,
    max_ma=np.inf,
    bibtex = {'Collaboration:2928094': '''@techreport{Collaboration:2928094,
      author        = "CMS Collaboration",
      collaboration = "CMS",
      title         = "{CMS flavor physics projections for the update of the
                       European Strategy for Particle Physics}",
      institution   = "CERN",
      reportNumber  = "CMS-NOTE-2025-004, CERN-CMS-NOTE-2025-004",
      address       = "Geneva",
      year          = "2025",
      url           = "https://cds.cern.ch/record/2928094",
}'''}
)

lhcbHL_Bstomumu = MeasurementConstant(
    'LHCb:2018roe',
    'flat',
    3.09e-9,
    0.16e-9,
    0.16e-9,
    max_ma=np.inf
)

lhcbHL_B0tomumu = MeasurementConstant(
    'LHCb:2018roe',
    'flat',
    1.06e-10,
    0.12e-10,
    0.12e-10,
    max_ma=np.inf
)

lhcb_Bstoee = MeasurementConstantBound(
    'LHCb:2020pcv',
    'flat',
    9.4e-9,
    max_ma=np.inf,
)

lhcb_B0toee = MeasurementConstantBound(
    'LHCb:2020pcv',
    'flat',
    2.5e-9,
    max_ma=np.inf,
)

lhcb_Bstotautau = MeasurementConstantBound(
    'LHCb:2017myy',
    'flat',
    5.2e-3,
    max_ma=np.inf,
)

lhcb_B0totautau = MeasurementConstantBound(
    'LHCb:2017myy',
    'flat',
    1.6e-3,
    max_ma=np.inf,
)

belle_Bstogammagamma = MeasurementConstantBound(
    'Belle:2014sac',
    'flat',
    3.1e-6,
    max_ma=np.inf,
)

babar_B0togammagamma = MeasurementConstantBound(
    'BaBar:2010qny',
    'flat',
    3.2e-7,
    max_ma=np.inf,
)

belle_D0togammagamma = MeasurementConstantBound(
    'Belle:2015pzk',
    'flat',
    8.5e-7,
    max_ma=np.inf,
)

belleII_D0togammagamma = MeasurementConstantBound(
    'Belle-II:2022cgf',
    'flat',
    1.5e-7,
    max_ma=np.inf,
)

belle_D0toee = MeasurementConstantBound(
    'Belle:2010ouj',
    'flat',
    7.9e-8,
    max_ma=np.inf,
)

lhcb_D0tomumu = MeasurementConstantBound(
    'LHCb:2022jaa',
    'flat',
    3.1e-9,
    max_ma=np.inf,
)

cms_D0tomumu = MeasurementConstantBound(
    'CMS:2025fmx',
    'flat',
    2.4e-9,
    max_ma=np.inf,
)

#LHCb Ds+->pi+ mu mu
    #Experiment: LHCb
    #arXiv: 1304.6365
def lhcb_Dsptopipmumu(x):
    citations.register_inspire('LHCb:2013hxr')
    q2min = [0.250, 1.250] #Dimuon mass
    q2max = [0.525, 2.000] #Dimuon mass
    value = [6.9e-8, 16.0e-8]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

lhcb_Dptopipmumu = MeasurementConstantBound(
    'LHCb:2020car',
    'prompt',
    67e-9,
    rmin = 150e-4,
    mass_parent=mDplus,
    mass_sibling=mpi0
)

lhcbHL_Dptopipmumu = MeasurementConstantBound(
    'LHCb:2018roe',
    'prompt',
    0.37e-8,
    rmin = 150e-4,
    mass_parent=mDplus,
    mass_sibling=mpi0
)

lhcb_DstoKpmumu = MeasurementConstantBound(
    'LHCb:2020car',
    'prompt',
    54e-9,
    rmin = 150e-4,
    mass_parent=mDs,
    mass_sibling=mK
)

e653_Dptorhopmumu = MeasurementConstantBound(
    'E653:1995rpz',
    'prompt',
    5.6e-4,
    rmin=1.5,
    mass_parent=mDplus,
    mass_sibling=mrho
)

e653_D0topi0mumu = MeasurementConstantBound(
    'E653:1995rpz',
    'prompt',
    1.8e-4,
    rmin=1.5,
    mass_parent=mD0,
    mass_sibling=mpi0
)

cleo_D0toetamumu = MeasurementConstantBound(
    'CLEO:1996jxx',
    'prompt',
    5.3e-4,
    rmin=4.7,
    mass_parent=mD0,
    mass_sibling=meta
)

e791_D0torho0mumu = MeasurementConstantBound(
    'E791:2000jkj',
    'prompt',
    2.2e-5,
    mass_parent=mD0,
    mass_sibling=mrho,
    rmin=1.5
)

pdg_KLtomumu = MeasurementConstant(
    ['ParticleDataGroup:2024cfk', 'E871:2000wvm', 'Akagi:1994bb', 'E791:1994xxb'],
    'flat',
    6.84e-9,
    0.11e-9,
    0.11e-9,
    max_ma=np.inf
)

e871_KLtoee = MeasurementConstant(
    'BNLE871:1998bii',
    'flat',
    8.7e-12,
    4.1e-12,
    5.7e-12,
    max_ma=np.inf
)

lhcb_KStomumu = MeasurementConstantBound(
    'LHCb:2020ycd',
    'flat',
    2.1e-10,
    max_ma=np.inf,
    conf_level=0.9
)

kloe_KStoee = MeasurementConstantBound(
    'KLOE:2008acb',
    'flat',
    9e-9,
    max_ma=np.inf
)

############ Quarkonia decays ############
visible = "visible/"

#BaBar Y(2S, 3S)--> Hadrons
    #Experiment: BaBar
    #arXiv:1108.3549
    #Results at 90% confidence level
    #Branching ratio

babar_Y3S_hadrons = MeasurementInterpolatedBound(
    'BaBar:2011kau',
    os.path.join(current_dir, visible, 'Babar_BR_hadrons_binned.txt'),
    'prompt',
    rmin=2.0,
    lab_boost=0.469/(1-0.469**2)**0.5, #gamma = 0.469, correspinding to E_electron = 8.6GeV and E_positron = 3.1GeV
    mass_parent=mUpsilon3S,
    mass_sibling=0
)

babar_Y3S_mumu = MeasurementInterpolatedBound(
    'BaBar:2009lbr',
    os.path.join(current_dir, visible, 'babar_Y3S_mumu.txt'),
    'prompt',
    rmin=2.0,
    lab_boost=0.469/(1-0.469**2)**0.5, #gamma = 0.469, correspinding to E_electron = 8.6GeV and E_positron = 3.1GeV
    mass_parent=mUpsilon3S,
    mass_sibling=0
    )

#BaBar Y(1S)--> Muons
    #Experiment: BaBar
    #arXiv:1210.0287
    #Results at 90% confidence level
    #Branching ratio
babar_Y1s_mumu = MeasurementInterpolatedBound(
    'BaBar:2012wey',
    os.path.join(current_dir, visible, 'Babar_BR_mumu_binned.txt'),
    'prompt',
    rmin=2.0,
    lab_boost=0.469/(1-0.469**2)**0.5, #gamma = 0.469, correspinding to E_electron = 8.6GeV and E_positron = 3.1GeV
    mass_parent=mUpsilon1S,
    mass_sibling=0
)

#BaBar Y(1S)--> c c
    #Experiment: BaBar
    #arXiv:1502.06019
    #Results at 90% confidence level
    #Branching ratio
babar_Y1s_cc = MeasurementInterpolatedBound(
    'BaBar:2015cce',
    os.path.join(current_dir, visible, 'Babar_BR_cc_binned.txt'),
    'prompt',
    rmin=2.0,
    lab_boost=0.469/(1-0.469**2)**0.5, #gamma = 0.469, correspinding to E_electron = 8.6GeV and E_positron = 3.1GeV
    mass_parent=mUpsilon1S,
    mass_sibling=0
)

#Belle Y(1S)--> Leptons
    #Experiment: Belle
    #arXiv:2112.11852
    #Results at 90% confidence level
    #Branching ratio
belle_Y1S_mumu = MeasurementInterpolatedBound(
    'Belle:2021rcl',
    os.path.join(current_dir, visible, 'Belle_BR_mumu_binned.txt'),
    'prompt',
    lab_boost=0.42,
    mass_parent=mUpsilon1S,
    mass_sibling=0,
    rmin=4
)

belle_Y1S_tautau = MeasurementInterpolatedBound(
    'Belle:2021rcl',
    os.path.join(current_dir, visible, 'Belle_BR_tautau_binned.txt'),
    'prompt',
    lab_boost=0.42,
    mass_parent=mUpsilon1S,
    mass_sibling=0,
    rmin=4
)

#BESIII J/psi--> mu mu
    #Experiment: BESIII
    #arXiv:2109.12625
    #Results at 90% confidence level
    #Branching ratio
besIII_Jpsi_mumu = MeasurementInterpolatedBound(
    'BESIII:2021ges',
    os.path.join(current_dir, visible, 'BES_BR_mumu_binned.txt'),
    'prompt',
    rmin = rmax_besIII,
    mass_parent=mJpsi,
    mass_sibling=0,
    lab_boost=11e-3 #Symmetric beams crossing at angle theta=11mrad. pJ/psi = sqrt(s)*sin(theta) approx sqrt(s)*theta
)

besIII_Jpsi_3gamma = MeasurementInterpolatedBound(
    'BESIII:2024hdv',
    os.path.join(current_dir, visible, 'BESIII_Jpsi3photon.txt'),
    'prompt',
    rmin = rmax_besIII,
    mass_parent=mJpsi,
    mass_sibling=0,
    lab_boost=11e-3 #Symmetric beams crossing at angle theta=11mrad. pJ/psi = sqrt(s)*sin(theta) approx sqrt(s)*theta
)

belleII_Upsilon4S_3gamma = MeasurementInterpolatedBound(
    'Belle-II:2020jti',
    os.path.join(current_dir, visible, 'Belle2_gamma_binned.txt'),
    'prompt',
    rmin=0.1,
    lab_boost=0.28,
    mass_parent=10.58,
    conf_level=0.95,
    mass_sibling=0
)

belleII_Upsilon4S_gammatautau = MeasurementInterpolatedBound(
    'Alda:2024cxn',
    os.path.join(current_dir, visible, 'BelleII_gammatautau.txt'),
    'prompt',
    rmin=0.1,
    lab_boost=0.28,
    mass_parent=10.58,
    conf_level=0.95,
    mass_sibling=0
)


lhcb_bkmumu_displvertex = MeasurementDisplacedVertexBound('LHCb:2016awg', os.path.join(current_dir, visible, 'LHCb_BKmumu_displ.npy'), 0.95)

lhcb_bks0mumu_displvertex = MeasurementDisplacedVertexBound('LHCb:2015nkv', os.path.join(current_dir, visible, 'LHCb_BKsmumu_displ.npy'), 0.95)

charm_bkmumu_displvertex = MeasurementDisplacedVertexBound(['Dobrich:2018jyi', 'CHARM:1985anb'], os.path.join(current_dir, visible, 'CHARM_BKmumu_displ.npy'), 0.95)

na62proj_bkmumu_displvertex = MeasurementDisplacedVertexBound('Dobrich:2018jyi', os.path.join(current_dir, visible, 'NA62_BKmumu_displ.npy'), 0.95)

shipproj_bkmumu_displvertex = MeasurementDisplacedVertexBound('Dobrich:2018jyi', os.path.join(current_dir, visible, 'SHiP_BKmumu_displ.npy'), 0.95)

belleII_bkmumu_displvertex = MeasurementDisplacedVertexBound('Belle-II:2023ueh', os.path.join(current_dir, visible, 'belleII_BKmumu_displ.npy'), 0.95)

belleII_bks0mumu_displvertex = MeasurementDisplacedVertexBound('Belle-II:2023ueh', os.path.join(current_dir, visible, 'belleII_B0K0smumu_displ.npy'), 0.95)

belleII_bkee_displvertex = MeasurementDisplacedVertexBound('Belle-II:2023ueh', os.path.join(current_dir, visible, 'belleII_BKee_displ.npy'), 0.95)

belleII_bks0ee_displvertex = MeasurementDisplacedVertexBound('Belle-II:2023ueh', os.path.join(current_dir, visible, 'belleII_B0K0see_displ.npy'), 0.95)

babar_bkphotons_displvertex = MeasurementDisplacedVertexBound('BaBar:2021ich', os.path.join(current_dir, visible, 'babar_BKphotons_displ.npy'), 0.95)

babar_bktautau = MeasurementConstantBound('BaBar:2016wgb', 'prompt', 2.25e-3, min_ma = 2*mtau, conf_level=0.9, rmin =100, mass_parent=mB, mass_sibling=mK)

belleII_B0Kstautau = MeasurementConstantBound(
    'Belle-II:2022cgf',
    'prompt',
    1.6e-3,
    min_ma=2*mtau,
    rmin=0.1,
    rmax=100,
    lab_boost=0.28,
    mass_parent=mB0,
    mass_sibling=mKst0
)

belleII_bktautau = MeasurementConstantBound(
    'Belle-II:2018jsg',
    'prompt',
    2e-5,
    2*mtau,
    rmin=0.1,
    rmax=100,
    lab_boost=0.28,
    mass_parent=mB,
    mass_sibling=mK
)

belleII_bktaumu = MeasurementConstantBound(
    'Belle-II:2018jsg',
    'prompt',
    3.3e-6,
    mtau+mmu,
    rmin=0.1,
    rmax=100,
    lab_boost=0.28,
    mass_parent=mB,
    mass_sibling=mK
)

belleII_bktaue = MeasurementConstantBound(
    'Belle-II:2018jsg',
    'prompt',
    2.1e-6,
    mtau+me,
    rmin=0.1,
    rmax=100,
    lab_boost=0.28,
    mass_parent=mB,
    mass_sibling=mK,
)

belleII_B0tautau = MeasurementConstantBound(
    'Belle-II:2018jsg',
    'flat',
    9.6e-5
)

belleII_Bstautau = MeasurementConstantBound(
    'Belle-II:2018jsg',
    'flat',
    8.1e-4
)

lhcbHL_Bstotautau = MeasurementConstantBound(
    'LHCb:2018roe',
    'flat',
    5e-4,
    max_ma=np.inf
)

belle_B0toK0stautau = MeasurementConstantBound('Belle:2021ecr', 'prompt', 3.1e-3, conf_level=0.9, min_ma=2*mtau, mass_parent=mB0, mass_sibling=mKst0, rmin=100)

#belle_Y1S_tautau = MeasurementInterpolatedBound('Belle:2021rcl', os.path.join(current_dir, visible, 'Belle_BR_tautau_binned.txt'), 'prompt', conf_level=0.9, min_ma=2*mtau, lab_boost=0.42, mass_parent=mUpsilon1S, mass_sibling=0, rmin=100)
babar_Y3S_tautau = MeasurementInterpolatedBound('BaBar:2009oxm', os.path.join(current_dir, visible, 'babar_Y3S_tautau.txt'), 'prompt', conf_level=0.9, lab_boost=0.469/(1-0.469**2)**0.5, mass_parent=mUpsilon3S, mass_sibling=0, rmin=10)

na62_Ktopimumu = MeasurementDisplacedVertexBound(
    ['NA62:2025upx', 'NA62:2022qes'],
    os.path.join(current_dir, visible, 'na62_kpimumu.npy'),
    decay_type = 'displaced'
)

na62_Ktopigammagamma = MeasurementDisplacedVertexBound(
    ['NA62:2025upx', 'NA62:2023olg'],
    os.path.join(current_dir, visible, 'na62_kpigammagamma.npy'),
    decay_type = 'displaced'
)

na62na48_kpigammagamma = MeasurementBinned(
    '',
    os.path.join(current_dir, visible, 'na62na48_Kplus_piphotons'),
    'prompt',
    rmin = 14000,
    lab_boost = 75/mK,
    mass_parent = mK,
    mass_sibling = mpi_pm
    )

na482_Kpimumu = MeasurementDisplacedVertexBound(
    'NA482:2016sfh',
    os.path.join(current_dir, visible, 'na482_kpimumu.npy'),
    rmax = 14000,
    lab_boost = 75/mK,
    mass_parent = mK,
    mass_sibling = mpi_pm
    )

microboone_Kpiee = MeasurementDisplacedVertexBound(
    'MicroBooNE:2021sov',
    os.path.join(current_dir, visible, 'microboone_kpiee.npy'),
    conf_level= 0.95
)

e787_Ktopigammagamma = MeasurementDisplacedVertexBound(
    'E787:1997abk',
    os.path.join(current_dir, visible, 'e787_kpigamma.npy'),
    conf_level= 0.9
)

belle_bpKomega3pi = MeasurementConstantBound(
    ['Belle:2013nby', 'Chakraborty:2021wda', 'ParticleDataGroup:2024cfk'],
    'prompt',
    (4.5e-6+2*0.5e-6)*0.892, #[BR(B+->K+omega(782)pi+pi-pi0) at 2 sigma]*BR(omega->pi+pi-pi0)
    conf_level=0.95,
    rmin = 4,
    lab_boost = 0.425,
    mass_parent = mB,
    mass_sibling = mK,
    min_ma = 0.73,
    max_ma = 0.83
)

belle_b0Komega3pi = MeasurementConstantBound(
    ['Belle:2013nby', 'Chakraborty:2021wda', 'ParticleDataGroup:2024cfk'],
    'prompt',
    (6.9e-6+2*2**0.5*0.4e-6)*0.892, #[BR(B0->K0omega(782)pi+pi-pi0) at 2 sigma]*BR(omega->pi+pi-pi0)
    conf_level=0.95,
    rmin = 4,
    lab_boost = 0.425,
    mass_parent = mB,
    mass_sibling = mK,
    min_ma = 0.73,
    max_ma = 0.83
)

babar_BKetapipi = MeasurementInterpolatedBound(
    ['Chakraborty:2021wda', 'BaBar:2008rth'],
    os.path.join(current_dir, visible, 'babar_BKetapipi.txt'),
    'prompt',
    conf_level=0.95,
    lab_boost=0.469/(1-0.469**2)**0.5,
    rmin = 10,
    mass_parent = mB,
    mass_sibling = mK
)

babar_bptopiee = MeasurementConstantBound(
    'BaBar:2013qaj',
    'prompt',
    12.5e-8,
    mass_parent=mB,
    mass_sibling=mpi_pm,
    lab_boost=0.469/(1-0.469**2)**0.5,
    rmin = 10,
)

babar_b0topiee = MeasurementConstantBound(
    'BaBar:2013qaj',
    'prompt',
    8.4e-8,
    mass_parent=mB0,
    mass_sibling=mpi0,
    lab_boost=0.469/(1-0.469**2)**0.5,
    rmin = 10,
)

babar_bptopimumu = MeasurementConstantBound(
    'BaBar:2013qaj',
    'prompt',
    5.5e-8,
    mass_parent=mB,
    mass_sibling=mpi_pm,
    lab_boost=0.469/(1-0.469**2)**0.5,
    rmin = 10,
)

babar_b0topimumu = MeasurementConstantBound(
    'BaBar:2013qaj',
    'prompt',
    6.9e-8,
    mass_parent=mB0,
    mass_sibling=mpi0,
    lab_boost=0.469/(1-0.469**2)**0.5,
    rmin = 10,
)

belle_bptopiee = MeasurementConstantBound(
    'Belle:2008tjs',
    'prompt',
    8e-8,
    rmin = 4,
    lab_boost = 0.425,
    mass_parent=mB,
    mass_sibling=mpi_pm
)

belle_bptopimumu = MeasurementConstantBound(
    'Belle:2008tjs',
    'prompt',
    6.9e-8,
    rmin = 4,
    lab_boost = 0.425,
    mass_parent=mB,
    mass_sibling=mpi_pm
)

belle_b0topiee = MeasurementConstantBound(
    'Belle:2008tjs',
    'prompt',
    22.7e-8,
    rmin = 4,
    lab_boost = 0.425,
    mass_parent=mB0,
    mass_sibling=mpi0
)

belle_b0topimumu = MeasurementConstantBound(
    'Belle:2008tjs',
    'prompt',
    18.4e-8,
    rmin = 4,
    lab_boost = 0.425,
    mass_parent=mB0,
    mass_sibling=mpi0
)

kloe_KStogammagamma = MeasurementConstant(
    'KLOE:2007rta',
    'flat',
    2.26e-6,
    np.sqrt(0.12**2+0.06**2)*1e-6,
    np.sqrt(0.12**2+0.06**2)*1e-6,
    max_ma=np.inf
)

na48_KStogammagamma = MeasurementConstant(
    'Lai:2002sr',
    'flat',
    2.713e-6,
    np.sqrt(6.3**2+0.5**2)*1e-8,
    np.sqrt(6.3**2+0.5**2)*1e-8,
    max_ma=np.inf
)

pdg_deltamK = MeasurementConstant(
    'ParticleDataGroup:2024cfk',
    'flat',
    5.293e-3, #ps^-1
    0.009e-3, #ps^-1
    0.009e-3, #ps^-1
    max_ma=np.inf
)

pdg_epsK = MeasurementConstant(
    'ParticleDataGroup:2024cfk',
    'flat',
    2.228e-3,
    0.011e-3,
    0.011e-3,
    max_ma=np.inf
)

hflav_xD0 = MeasurementConstant(
    'HeavyFlavorAveragingGroupHFLAV:2024ctg',
    'flat',
    0.407*1e-2,
    0.044*1e-2,
    0.044*1e-2,
    max_ma=np.inf
)

hflav_phi12D0 = MeasurementConstant(
    'HeavyFlavorAveragingGroupHFLAV:2024ctg',
    'flat',
    0.65*np.pi/180,
    0.90*np.pi/180,
    0.92*np.pi/180,
    max_ma=np.inf
)

hflav_ASL_B0 = MeasurementConstant(
    'HeavyFlavorAveragingGroupHFLAV:2024ctg',
    'flat',
    -0.0021,
    0.0017,
    0.0017,
    max_ma=np.inf
)


hflav_ASL_Bs = MeasurementConstant(
    'HeavyFlavorAveragingGroupHFLAV:2024ctg',
    'flat',
    -0.0006,
    0.0028,
    0.0028,
    max_ma=np.inf
)

belleII_deltaMd = MeasurementConstant(
    'Belle-II:2023bps',
    'flat',
    0.516,
    (0.008**2+0.005**2)**0.5,
    (0.008**2+0.005**2)**0.5,
    max_ma=np.inf
)

lhcb_deltaMs = MeasurementConstant(
    'LHCb:2023sim',
    'flat',
    17.743,
    (0.033**2+0.009**2)**0.5,
    (0.033**2+0.009**2)**0.5,
    max_ma=np.inf
)

belleII_taueinv = MeasurementInterpolatedBound(
    'Belle-II:2022heu',
    os.path.join(current_dir, invisible, 'BelleII_tau_e.txt'),
    'invisible',
    rmax=100,
    mass_parent=mtau,
    mass_sibling=me,
    lab_boost=0.28
)

belleII_taumuinv = MeasurementInterpolatedBound(
    'Belle-II:2022heu',
    os.path.join(current_dir, invisible, 'BelleII_tau_mu.txt'),
    'invisible',
    rmax=100,
    mass_parent=mtau,
    mass_sibling=mmu,
    lab_boost=0.28
)

belle_tau3e = MeasurementConstantBound(
    'Hayasaka:2010np',
    'prompt',
    2.7e-8,
    rmin=3.0,
    mass_parent=mtau,
    mass_sibling=me,
    lab_boost=0.42
)

belle_tau3mu = MeasurementConstantBound(
    'Hayasaka:2010np',
    'prompt',
    2.1e-8,
    rmin=3.0,
    mass_parent=mtau,
    mass_sibling=mmu,
    lab_boost=0.42
)

belle_tauemumu = MeasurementConstantBound(
    'Hayasaka:2010np',
    'prompt',
    2.7e-8,
    rmin=3.0,
    mass_parent=mtau,
    mass_sibling=me,
    lab_boost=0.42
)

belle_taumuee = MeasurementConstantBound(
    'Hayasaka:2010np',
    'prompt',
    1.8e-8,
    rmin=3.0,
    mass_parent=mtau,
    mass_sibling=mmu,
    lab_boost=0.42
)

babar_tauegammagamma = MeasurementConstantBound(
    'Bryman:2021ilc',
    'prompt',
    2.5e-4,
    rmin=3.0,
    mass_parent=mtau,
    mass_sibling=me,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_taumugammagamma = MeasurementConstantBound(
    'Bryman:2021ilc',
    'prompt',
    5.8e-4,
    rmin=3.0,
    mass_parent=mtau,
    mass_sibling=me,
    lab_boost=0.469/(1-0.469**2)**0.5
)

belleII_tau_3e = MeasurementConstantBound(
    'Belle-II:2022cgf',
    'prompt',
    5.088e-10,
    rmin=0.1,
    rmax=100,
    mass_parent=mtau,
    mass_sibling=me,
    lab_boost=0.28
)

belleII_tau_muee = MeasurementConstantBound(
    'Belle-II:2022cgf',
    'prompt',
    3.20e-10,
    rmin=0.1,
    rmax=100,
    mass_parent=mtau,
    mass_sibling=mmu,
    lab_boost=0.28
)

belleII_tau_emumu = MeasurementConstantBound(
    'Belle-II:2022cgf',
    'prompt',
    4.87e-10,
    rmin=0.1,
    rmax=100,
    mass_parent=mtau,
    mass_sibling=me,
    lab_boost=0.28
)

belleII_tau_3mu = MeasurementConstantBound(
    'Belle-II:2022cgf',
    'prompt',
    3.93e-10,
    rmin=0.1,
    rmax=100,
    mass_parent=mtau,
    mass_sibling=mmu,
    lab_boost=0.28
)

twist_mueinv = MeasurementInterpolatedBound(
    'TWIST:2014ymv',
    os.path.join(current_dir, invisible, 'TWIST_muea.txt'),
    'invisible',
    rmax = 16.5,
    mass_parent=mmu,
    mass_sibling=me,
    lab_boost=0.0
)

sindrum_mu3e = MeasurementConstantBound(
    'SINDRUM:1987nra',
    'prompt',
    1e-12,
    rmin=100,
    mass_parent=mmu,
    mass_sibling=me,
    lab_boost=0.0
)

cristalbox_muegammagamma = MeasurementConstantBound(
    'Bolton:1988af',
    'prompt',
    7.2e-11,
    rmin=100,
    mass_parent=mmu,
    mass_sibling=me,
    lab_boost=0.0
)

babar_BplusKtaumu = MeasurementConstantBound(
    'BaBar:2012azg',
    'prompt',
    4.8e-5,
    conf_level=0.9,
    rmin=10,
    mass_parent=mB,
    mass_sibling=mK,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_BplusKtaue = MeasurementConstantBound(
    'BaBar:2012azg',
    'prompt',
    3.0e-5,
    conf_level=0.9,
    rmin=10,
    mass_parent=mB,
    mass_sibling=mK,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_Bpluspitaumu = MeasurementConstantBound(
    'BaBar:2012azg',
    'prompt',
    7.2e-5,
    conf_level=0.9,
    rmin=10,
    mass_parent=mB,
    mass_sibling=mpi_pm,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_Bpluspitaue = MeasurementConstantBound(
    'BaBar:2012azg',
    'prompt',
    4.8e-5,
    conf_level=0.9,
    rmin=10,
    mass_parent=mB,
    mass_sibling=mpi_pm,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_BplusKmue = MeasurementConstantBound(
    'BaBar:2006tnv',
    'prompt',
    9.1e-8,
    rmin = 10,
    mass_parent=mB,
    mass_sibling=mK,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_B0Kmue = MeasurementConstantBound(
    'BaBar:2006tnv',
    'prompt',
    27e-8,
    rmin = 10,
    mass_parent=mB0,
    mass_sibling=mKL,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_BplusKstmue = MeasurementConstantBound(
    'BaBar:2006tnv',
    'prompt',
    140e-8,
    rmin = 10,
    mass_parent=mB,
    mass_sibling=mKst0,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_B0Kstmue = MeasurementConstantBound(
    'BaBar:2006tnv',
    'prompt',
    58e-8,
    rmin = 10,
    mass_parent=mB0,
    mass_sibling=mKst0,
    lab_boost=0.469/(1-0.469**2)**0.5
)

lhcb_B0Ksttaumu = MeasurementConstantBound(
    'LHCb:2022wrs',
    'prompt',
    1.82e-5,
    rmin = 150e-4,
    mass_parent=mB0,
    mass_sibling=mKst0
)

lhcb_DstoKmue = MeasurementConstantBound(
    'LHCb:2020car',
    'prompt',
    (790+560)*1e-9,
    rmin = 150e-4,
    mass_parent=mDs,
    mass_sibling=mK
)

lhcb_Dplustopimue = MeasurementConstantBound(
    'LHCb:2020car',
    'prompt',
    (230+220)*1e-9,
    rmin = 150e-4,
    mass_parent=mDplus,
    mass_sibling=mpi_pm
)

e653_DstoKstmue = MeasurementConstantBound(
    'E653:1995rpz',
    'prompt',
    1.4e-3,
    rmin=1.5,
    mass_parent=mDs,
    mass_sibling=mKst0
)

babar_D0pi0mue = MeasurementConstantBound(
    'BaBar:2020faa',
    'prompt',
    8e-7,
    rmin=10,
    mass_parent=mD0,
    mass_sibling=mpi0,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_D0etamue = MeasurementConstantBound(
    'BaBar:2020faa',
    'prompt',
    22.5e-7,
    rmin=10,
    mass_parent=mD0,
    mass_sibling=meta,
    lab_boost=0.469/(1-0.469**2)**0.5
)

babar_D0rhomue = MeasurementConstantBound(
    'BaBar:2020faa',
    'prompt',
    5e-7,
    rmin=10,
    mass_parent=mD0,
    mass_sibling=mrho,
    lab_boost=0.469/(1-0.469**2)**0.5
)

na62_Kpluspimue = MeasurementConstantBound(
    'NA62:2021zxl',
    'prompt',
    2*6.6e-11,
    rmin = 14000,
    lab_boost = 75/mK,
    mass_parent = mK,
    mass_sibling = mpi_pm
    )

ktev_KLtopi0mue = MeasurementConstantBound(
    'KTeV:2007cvy',
    'prompt',
    0.76e-10,
    rmin = 10500,
    lab_boost = 70/mK,
    mass_parent = mKL,
    mass_sibling = mpi0
)

belle_Y1S_mue = MeasurementConstantBound(
    'Belle:2022cce',
    'prompt',
    4.2e-7,
    lab_boost=0.42,
    mass_parent=mUpsilon1S,
    mass_sibling=0,
    rmin=4
)

belle_Y1S_taue = MeasurementConstantBound(
    'Belle:2022cce',
    'prompt',
    6.5e-6,
    lab_boost=0.42,
    mass_parent=mUpsilon1S,
    mass_sibling=0,
    rmin=4
)

belle_Y1S_taumu = MeasurementConstantBound(
    'Belle:2022cce',
    'prompt',
    6.1e-6,
    lab_boost=0.42,
    mass_parent=mUpsilon1S,
    mass_sibling=0,
    rmin=4
)

belleII_tau3mu = MeasurementConstantBound(
    'Belle-II:2024sce',
    'prompt',
    1.9e-8,
    rmin=0.1,
    lab_boost=0.28,
    mass_parent=mtau,
    conf_level=0.95,
    mass_sibling=mmu
)

belleII_B0KStaumu = MeasurementConstantBound(
    'Belle-II:2024qod',
    'prompt',
    4.7e-5,
    rmin=0.1,
    lab_boost=0.28,
    mass_parent=mB0,
    conf_level=0.95,
    mass_sibling=mKL
)

belleII_B0KStaue = MeasurementConstantBound(
    'Belle-II:2024qod',
    'prompt',
    2.3e-5,
    rmin=0.1,
    lab_boost=0.28,
    mass_parent=mB0,
    conf_level=0.95,
    mass_sibling=mKL
)

dw_Bplus = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.B_plus.width/1000,
    max_ma = mB,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.B_plus.width_upper/particles.B_plus.width)
)

dw_B0 = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.B_0.width/1000,
    max_ma = mB0,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.B_0.width_upper/particles.B_0.width)
)

dw_D0 = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.D_0.width/1000,
    max_ma = mD0,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.D_0.width_upper/particles.D_0.width)
)

dw_Dplus = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.D_plus.width/1000,
    max_ma = mDplus,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.D_plus.width_upper/particles.D_plus.width)
)

dw_Dsplus = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.D_s_plus.width/1000,
    max_ma = mDs,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.D_s_plus.width_upper/particles.D_s_plus.width)
)

dw_Kplus = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.K_plus.width/1000,
    max_ma = mK,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.K_plus.width_upper/particles.K_plus.width)
)

dw_KL = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.K_L_0.width/1000,
    max_ma = mKL,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.K_L_0.width_upper/particles.K_L_0.width)
)

dw_KS = MeasurementConstantBound(
    'ParticleDataGroup:2024cfk',
    'flat',
    particles.K_S_0.width/1000,
    max_ma = mKL,
    conf_level= 1/(1+np.sqrt(np.pi/2)*particles.K_S_0.width_upper/particles.K_S_0.width)
)

def get_measurements(process: str | tuple, exclude_projections: bool = True) -> dict[str, MeasurementBase]:
    """Retrieve measurements based on the given transition.

    Parameters
    ----------
    transition : str
        The particle transition in the format 'initial -> final'.

    exclude_projections : bool
        Flag to exclude projection measurements. Defaults to True.

    Returns
    -------
    measurements : dict[str, MeasurementBase]
        A dictionary mapping experiment names to their corresponding measurement data.

    Raises
    ------
    KeyError
        If no measurements are found for the given transition.
    """

    if process == 'delta_mK0':
        return {'PDG': pdg_deltamK}
    elif process == 'epsK':
        return {'PDG': pdg_epsK}
    elif process == 'x_D0':
        return {'HFLAV': hflav_xD0}
    elif process == 'phi12_D0':
        return {'HFLAV': hflav_phi12D0}
    elif process == 'delta_mB0':
        return {'Belle II': belleII_deltaMd}
    elif process == 'delta_mBs':
        return {'LHCb': lhcb_deltaMs}
    elif process == 'ASL_B0':
        return {'HFLAV': hflav_ASL_B0}
    elif process == 'ASL_Bs':
        return {'HFLAV': hflav_ASL_Bs}
    elif process in ['B+', 'B-']:
        return {'PDG': dw_Bplus}
    elif process in ['B0', 'Bd0']:
        return {'PDG': dw_B0}
    elif process == 'D0':
        return {'PDG': dw_D0}
    elif process in ['D+', 'D-']:
        return {'PDG': dw_Dplus}
    elif process in ['Ds+', 'Ds-']:
        return {'PDG': dw_Dsplus}
    elif process in ['K+', 'K-']:
        return {'PDG': dw_Kplus}
    elif process in ['KL', 'K0L']:
        return {'PDG': dw_KL}
    elif process in ['KS', 'K0S']:
        return {'PDG': dw_KS}

    if isinstance(process, str):
        transition = process
    else:
        transition = process[0]
        args = process[1:]

    initial, final = parse(transition)
    #Initial state B+
    if initial == ['B+'] and final == sorted(['K+', 'alp']):
        return {'Belle II': belleII_bptoknunu_lightmediator,
                'BaBar': babar_btoknunu_lightmediator
                #,'BaBar + Belle II': combined_btoknunu_lightmediator
                }
    elif initial == ['B+'] and final == sorted(['K*+', 'alp']):
        return {'BaBar': babar_btokstarnunu_lightmediator}
    elif initial == ['B+'] and final == sorted(['pion+', 'alp']):
        return {'Belle': belle_Bchargedtopichargednunu}
    elif initial == ['B+'] and final == sorted(['rho+', 'alp']):
        return {'Belle': belle_Bchargedtorhochargednunu}
    elif initial == ['B+'] and final == sorted(['pion+', 'electron', 'electron']):
        return {'BaBar': babar_bptopiee, 'Belle': belle_bptopiee}
    elif initial == ['B+'] and final == sorted(['pion+', 'muon', 'muon']):
        return {'BaBar': babar_bptopimumu, 'Belle': belle_bptopimumu}
    elif initial == ['B+'] and final == sorted(['pion+', 'electron', 'tau']):
        return {'BaBar': babar_Bpluspitaue}
    elif initial == ['B+'] and final == sorted(['pion+', 'muon', 'tau']):
        return {'BaBar': babar_Bpluspitaumu}
    elif initial == ['B+'] and final == sorted(['K+', 'electron', 'electron']):
        return {'Belle II': belleII_bkee_displvertex}
    elif initial == ['B+'] and final == sorted(['K+', 'muon', 'muon']):
        if exclude_projections:
            return {'LHCb': lhcb_bkmumu_displvertex, 'Belle II': belleII_bkmumu_displvertex, 'CHARM': charm_bkmumu_displvertex}
        else:
            return {'LHCb': lhcb_bkmumu_displvertex, 'Belle II': belleII_bkmumu_displvertex, 'CHARM': charm_bkmumu_displvertex, 'NA62': na62proj_bkmumu_displvertex, 'SHiP': shipproj_bkmumu_displvertex}  
    elif initial == ['B+'] and final == sorted(['K+', 'tau', 'tau']):
        if exclude_projections:
            return {'BaBar': babar_bktautau}
        else:
            return {'BaBar': babar_bktautau, 'Belle II 50ab-1': belleII_bktautau}
    elif initial == ['B+'] and final == sorted(['K+', 'photon', 'photon']):
        return {'BaBar': babar_bkphotons_displvertex}
    elif initial == ['B+'] and final == sorted(['K+', 'pion+', 'pion-', 'pion0']):
        return {'Belle': belle_bpKomega3pi}
    elif initial == ['B+'] and final == sorted(['K+', 'eta', 'pion+', 'pion-']):
        return {'BaBar': babar_BKetapipi}
    elif initial == ['B+'] and final == sorted(['K+', 'muon', 'tau']):
        if exclude_projections:
            return {'BaBar': babar_BplusKtaumu}
        else:
            return {'BaBar': babar_BplusKtaumu, 'Belle II 50ab-1': belleII_bktaumu}
    elif initial == ['B+'] and final == sorted(['K+', 'electron', 'tau']):
        if exclude_projections:
            return {'BaBar': babar_BplusKtaue}
        else:
            return {'BaBar': babar_BplusKtaue, 'Belle II 50ab-1': belleII_bktaue}
    elif initial == ['B+'] and final == sorted(['K+', 'muon', 'electron']):
        return {'BaBar': babar_BplusKmue}
    elif initial == ['B+'] and final == sorted(['K*+', 'muon', 'electron']):
        return {'BaBar': babar_BplusKstmue}
    #Initial state B0
    elif initial == ['B0'] and final == sorted(['pion0', 'alp']):
        return {'Belle': belle_B0topi0nunu}
    elif initial == ['B0'] and final == sorted(['K0', 'alp']):
        return {'Belle': belle_B0toK0nunu}
    elif initial == ['B0'] and final == sorted(['rho0', 'alp']):
        return {'Belle': belle_B0torho0nunu}
    elif initial == ['B0'] and final == sorted(['pion0', 'electron', 'electron']):
        return {'BaBar': babar_b0topiee, 'Belle': belle_b0topiee}
    elif initial == ['B0'] and final == sorted(['pion0', 'muon', 'muon']):
        return {'BaBar': babar_b0topimumu, 'Belle': belle_b0topimumu}
    elif initial == ['B0'] and final == sorted(['K*0', 'electron', 'electron']):
        return {'Belle II': belleII_bks0ee_displvertex}
    elif initial == ['B0'] and final == sorted(['K*0', 'muon', 'muon']):
        return {'LHCb': lhcb_bks0mumu_displvertex, 'Belle II': belleII_bks0mumu_displvertex}
    elif initial == ['B0'] and final == sorted(['K*0', 'tau', 'tau']):
        if exclude_projections:
            return {'Belle': belle_B0toK0stautau}
        else:
            return {'Belle': belle_B0toK0stautau, 'Belle II 50ab-1': belleII_B0Kstautau}
    elif initial == ['B0'] and final == sorted(['K0', 'pion+', 'pion-', 'pion0']):
        return {'Belle': belle_b0Komega3pi}
    elif initial == ['B0'] and final == ['electron', 'electron']:
        return {'LHCb': lhcb_B0toee}
    elif initial == ['B0'] and final == ['muon', 'muon']:
        if exclude_projections:
            return {'LHCb': lhcb_B0tomumu, 'CMS': cms_B0tomumu}
        else:
            return {'LHCb': lhcb_B0tomumu, 'CMS': cms_B0tomumu, 'ATLAS HL-LHC': atlasHL_B0tomumu, 'CMS HL-LHC': cmsHL_B0tomumu, 'LHCb HL-LHC': lhcbHL_B0tomumu}
    elif initial == ['B0'] and final == sorted(['tau', 'tau']):
        if exclude_projections:
            return {'LHCb': lhcb_B0totautau}
        else:
            return {'LHCb': lhcb_B0totautau, 'Belle II 50ab-1': belleII_B0tautau}
        return {'LHCb': lhcb_B0totautau}
    elif initial == ['B0'] and final == sorted(['photon', 'photon']):
        return {'BaBar': babar_B0togammagamma}
    elif initial == ['B0'] and final == sorted(['K0', 'muon', 'electron']):
        return {'BaBar': babar_B0Kmue}
    elif initial == ['B0'] and final == sorted(['K*0', 'muon', 'electron']):
        return {'BaBar': babar_B0Kstmue}
    elif initial == ['B0'] and final == sorted(['K*0', 'muon', 'tau']):
        return {'LHCb': lhcb_B0Ksttaumu}
    #Initial state Bs
    elif initial == ['Bs'] and final == sorted(['phi', 'alp']):
        return {'DELPHI': delphi_Bstophinunu}
    elif initial == ['Bs'] and final == ['electron', 'electron']:
        return {'LHCb': lhcb_Bstoee}
    elif initial == ['Bs'] and final == sorted(['muon', 'muon']):
        if exclude_projections:
            return {'LHCb': lhcb_Bstomumu, 'CMS': cms_Bstomumu}
        else:
            return {'LHCb': lhcb_Bstomumu, 'CMS': cms_Bstomumu, 'ATLAS HL-LHC': atlasHL_Bstomumu, 'CMS HL-LHC': cmsHL_Bstomumu, 'LHCb HL-LHC': lhcbHL_Bstomumu}
    elif initial == ['Bs'] and final == sorted(['tau', 'tau']):
        if exclude_projections:
            return {'LHCb': lhcb_Bstotautau}
        else:
            return {'LHCb': lhcb_Bstotautau, 'Belle II 5ab-1': belleII_Bstautau, 'LHCb HL-LHC': lhcbHL_Bstotautau}
    elif initial == ['Bs'] and final == sorted(['photon', 'photon']):
        return {'Belle': belle_Bstogammagamma}
    #Initial state J/psi
    elif initial == ['J/psi'] and final == sorted(['photon', 'alp']):
        return {'BESIII': besIII_Jpsiinv}
    elif initial == ['J/psi'] and final == ['photon', 'photon', 'photon']:
        return {'BESIII': besIII_Jpsi_3gamma}
    elif initial == ['J/psi'] and final == sorted(['muon', 'muon', 'photon']):
        return {'BESIII': besIII_Jpsi_mumu}
    #Initial state Y(1S)
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'alp']):
        return {'Belle': belle_Y1S_inv}
    elif initial == ['Upsilon(1S)'] and final == sorted(['muon', 'muon', 'photon']):
        return {'BaBar': babar_Y1s_mumu, 'Belle': belle_Y1S_mumu}
    elif initial == ['Upsilon(1S)'] and final == sorted(['tau', 'tau', 'photon']):
        return {'Belle': belle_Y1S_tautau}
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'charm', 'charm']):
        return {'BaBar': babar_Y1s_cc}
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'electron', 'muon']):
        return {'Belle': belle_Y1S_mue}
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'tau', 'electron']):
        return {'Belle': belle_Y1S_taue}
    elif initial == ['Upsilon(1S)'] and final == sorted(['photon', 'tau', 'muon']):
        return {'Belle': belle_Y1S_taumu}
    #Initial state Upsilon(3S)
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'alp']):
        return {'BaBar': babar_Y3S_inv}
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'tau', 'tau']):
        return {'BaBar': babar_Y3S_tautau}
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'muon', 'muon']):
        return {'BaBar': babar_Y3S_mumu}
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'hadrons']):
        return {'BaBar': babar_Y3S_hadrons}
    #Non-resonant e+e- -> gamma a @ mUpsilon(4S)
    elif initial == ['electron', 'electron'] and final == ['photon', 'photon', 'photon'] and len(args)==1 and args[0] == 10.58**2:
        return {'Belle II': belleII_Upsilon4S_3gamma}
    elif initial == ['electron', 'electron'] and final == ['photon', 'tau', 'tau'] and len(args)==1 and args[0] == 10.58**2:
        if exclude_projections:
            raise KeyError(f"No measurements for {transition}")
        else:
            return {'Belle II': belleII_Upsilon4S_gammatautau}
    #initial state D0
    elif initial == ['D0'] and final == sorted(['pion0', 'alp']):
        return {'BESIII': besIII_D0topi0nunu}
    elif initial == ['D0'] and final == sorted(['pion0', 'electron', 'electron']):
        return {'BESIII': besIII_D0topi0ee}
    elif initial == ['D0'] and final == sorted(['pion0', 'muon', 'muon']):
        return {'E653': e653_D0topi0mumu}
    elif initial == ['D0'] and final == sorted(['eta', 'electron', 'electron']):
        return {'BESIII': besIII_D0toetaee}
    elif initial == ['D0'] and final == sorted(['eta', 'muon', 'muon']):
        return {'CLEO II': cleo_D0toetamumu}
    elif initial == ['D0'] and final == sorted(['rho0', 'electron', 'electron']):
        return {'E791': e791_D0torho0ee}
    elif initial == ['D0'] and final == sorted(['rho0', 'muon', 'muon']):
        return {'E791': e791_D0torho0mumu}
    elif initial == ['D0'] and final == sorted(['photon', 'photon']):
        if exclude_projections:
            return {'Belle': belle_D0togammagamma}
        else:
            return {'Belle': belle_D0togammagamma, 'Belle II 50ab-1': belleII_D0togammagamma}
    elif initial == ['D0'] and final == sorted(['electron', 'electron']):
        return {'Belle': belle_D0toee}
    elif initial == ['D0'] and final == sorted(['muon', 'muon']):
        return {'LHCb': lhcb_D0tomumu, 'CMS': cms_D0tomumu}
    elif initial == ['D0'] and final == sorted(['pion0', 'muon', 'electron']):
        return {'BaBar': babar_D0pi0mue}
    elif initial == ['D0'] and final == sorted(['eta', 'muon', 'electron']):
        return {'BaBar': babar_D0etamue}
    elif initial == ['D0'] and final == sorted(['rho0', 'muon', 'electron']):
        return {'BaBar': babar_D0rhomue}
    #Initial state D+
    elif initial == ['D+'] and final == sorted(['pion+', 'electron', 'electron']):
        return {'LHCb': lhcb_Dptopipee}
    elif initial == ['D+'] and final == sorted(['pion+', 'muon', 'muon']):
        if exclude_projections:
            return {'LHCb': lhcb_Dptopipmumu}
        else:
            return {'LHCb': lhcb_Dptopipmumu, 'LHCb HL-LHC': lhcbHL_Dptopipmumu}
    elif initial == ['D+'] and final == sorted(['rho+', 'muon', 'muon']):
        return {'E653': e653_Dptorhopmumu}
    elif initial == ['D+'] and final == sorted(['pion+', 'electron', 'muon']):
        return {'LHCb': lhcb_Dplustopimue}
    #Initial state Ds+
    elif initial == ['Ds+'] and final == sorted(['K+', 'electron', 'electron']):
        return {'LHCb': lhcb_DstoKpee}
    elif initial == ['Ds+'] and final == sorted(['K+', 'muon', 'muon']):
        return {'LHCb': lhcb_DstoKpmumu}
    elif initial == ['Ds+'] and final == sorted(['K+', 'muon', 'electron']):
        return {'LHCb': lhcb_DstoKmue}
    elif initial == ['Ds+'] and final == sorted(['K*+', 'muon', 'electron']):
        return {'E653': e653_DstoKstmue}
    #Initial state K+
    elif initial == ['K+'] and final == sorted(['pion+', 'alp']):
        return {'E949': e949_Ktopiinv,
                'NA62': na62_Ktopiinv_2025,
                'NA62 (pi0)': na62_Ktopipi0}
    elif initial == ['K+'] and final == sorted(['pion+', 'electron', 'muon']):
        return {'NA62': na62_Kpluspimue}
    elif initial == ['K+'] and final == sorted(['pion+', 'photon', 'photon']):
        return {'NA62': na62_Ktopigammagamma}
    elif initial == ['K+'] and final == sorted(['muon', 'muon', 'pion+']):
        return {'NA48/2': na482_Kpimumu,
                'NA62': na62_Ktopimumu,}
    elif initial == ['K+'] and final == sorted(['electron', 'electron', 'pion+']):
        return {'MicroBooNE': microboone_Kpiee}
    elif initial == ['K+'] and final == sorted(['photon', 'photon', 'pion+']):
        return {'E787': e787_Ktopigammagamma}
    #Initial state KL
    elif initial == ['KL'] and final == sorted(['pion0', 'alp']):
        return {'KOTO': koto_kltopi0inv}
    elif initial == ['KL'] and final == ['electron', 'electron']:
        return {'E871': e871_KLtoee}
    elif initial == ['KL'] and final == ['muon', 'muon']:
        return {'PDG': pdg_KLtomumu}
    elif initial == ['KL'] and final == sorted(['pion0', 'muon', 'electron']):
        return {'KTeV': ktev_KLtopi0mue}
    #Initial state KS
    elif initial == ['KS'] and final == ['electron', 'electron']:
        return {'KLOE': kloe_KStoee}
    elif initial == ['KS'] and final == ['muon', 'muon']:
        return {'LHCb': lhcb_KStomumu}
    elif initial == ['KS'] and final == sorted(['photon', 'photon']):
        return {'NA48': na48_KStogammagamma, 'KLOE': kloe_KStogammagamma}
    #Lepton LFV decays
    elif initial == ['tau'] and final == sorted(['electron', 'alp']):
        return {'Belle II': belleII_taueinv}
    elif initial == ['tau'] and final == sorted(['muon', 'alp']):
        return {'Belle II': belleII_taumuinv}
    elif initial == ['tau'] and final == sorted(['electron', 'electron', 'electron']):
        if exclude_projections:
            return {'Belle': belle_tau3e}
        else:
            return {'Belle': belle_tau3e, 'Belle II 50ab-1': belleII_tau_3e}
    elif initial == ['tau'] and final == sorted(['muon', 'muon', 'muon']):
        if exclude_projections:
            return {'Belle': belle_tau3mu, 'Belle II': belleII_tau3mu}
        else:
            return {'Belle': belle_tau3mu, 'Belle II': belleII_tau3mu, 'Belle II 50ab-1': belleII_tau_3mu}
    elif initial == ['tau'] and final == sorted(['electron', 'muon', 'muon']):
        if exclude_projections:
            return {'Belle': belle_tauemumu}
        else:
            return {'Belle': belle_tauemumu, 'Belle II 50ab-1': belleII_tau_emumu}
    elif initial == ['tau'] and final == sorted(['muon', 'electron', 'electron']):
        if exclude_projections:
            return {'Belle': belle_taumuee}
        else:
            return {'Belle': belle_taumuee, 'Belle II 50ab-1': belleII_tau_muee}
    elif initial == ['tau'] and final == sorted(['electron', 'photon', 'photon']):
        return {'BaBar': babar_tauegammagamma}
    elif initial == ['tau'] and final == sorted(['muon', 'photon', 'photon']):
        return {'BaBar': babar_taumugammagamma}
    elif initial == ['muon'] and final == sorted(['electron', 'alp']):
        return {'TWIST': twist_mueinv}
    elif initial == ['muon'] and final == sorted(['electron', 'electron', 'electron']):
        return {'SINDRUM': sindrum_mu3e}
    elif initial == ['muon'] and final == sorted(['electron', 'photon', 'photon']):
        return {'Cristal Box': cristalbox_muegammagamma}
    else:
        raise KeyError(f"No measurements for {transition}")
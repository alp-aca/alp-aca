#File with all possible experimental measurement
import os
import numpy as np
from scipy.stats import chi2
from ..citations import citations
from ..constants import mUpsilon3S
from .classes import MeasurementBase, MeasurementConstantBound, MeasurementInterpolatedBound, MeasurementInterpolated, MeasurementDisplacedVertexBound, MeasurementBinned, rmax_belle, rmax_besIII, MeasurementConstant
from ..decays.particles import particle_aliases
from ..decays.decays import parse
from ..constants import mB, mB0, mBs, mK, mtau, mKst0, mKL, mpi0, mpi_pm, mphi, mZ
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

na62_Ktopiinv = MeasurementDisplacedVertexBound(
    'NA62:2020pwi',
    os.path.join(current_dir, invisible, 'na62_kpiInv.npy'),
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

babar_btoksnunu_lightmediator = MeasurementInterpolated(['Altmannshofer:2023hkn', 'BaBar:2013npw'], os.path.join(current_dir, invisible, 'Babar_BtoK_bestfit.txt'), 'invisible', rmax=50, lab_boost=0.469/(1-0.469**2)**0.5, mass_parent=mB, mass_sibling=mK)


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
def babar_upsilon3S(x):
    citations.register_inspire('BaBar:2008aby') 
    data_file_path = os.path.join(current_dir, invisible, 'Babar_BR_Y3S_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars

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
def belle_upsilon1S(x):
    citations.register_inspire('Belle:2018pzt') 
    data_file_path = os.path.join(current_dir, invisible, 'Belle_BR_Y1S_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars

#BesIII J/psi invisible
    #Experiment: BESIII
    #arXiv:2003.05594
    #Branching ratio
def besIII_Jpsiinv(x):
    citations.register_inspire('BESIII:2020sdo') 
    data_file_path = os.path.join(current_dir, invisible, 'BESIII_BR_Jpsi_inv_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars


#BesIII J/psi visible
    #Experiment: BESIII
    #arXiv:2211.12699
    #Branching ratio
def besIII_Jpsivis(x):
    citations.register_inspire('BESIII:2022rzz')
    data_file_path = os.path.join(current_dir, invisible, 'BESIII_BR_Jpsi_vis_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars


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
    rmax=rmax_besIII
)

#BelleII e+e- -> gamma a
    #Experiment: BelleII
    #arXiv: 2007.13071
    #@95% confidence level
    #Cross section (pb)
def belleII_upsilon4s(x):
    citations.register_inspire('Belle-II:2020jti')
    data_file_path = os.path.join(current_dir, invisible, 'Belle2_gamma_binned.txt')
    q2min, q2max, value = data_reading(data_file_path)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, 0.5*np.array(value), 0.5*np.array(value))
    return values*1e12, sigmals*1e12, sigmars*1e12

#################################### VISIBLE SEARCHES ####################################

###### Decays to gamma gamma ######

#NA62  K+->pi+ gamma gamma
    #Experiment: NA62
    #arXiv: 1402.4334
def na62_Ktopigammagamma(x):
    citations.register_inspire('NA62:2014ybm')
    q2min = [0.220] #Digamma momentum
    q2max = [0.354] #Digamma momentum
    value = [9.65e-7]
    sigmal = [0.63e-7]
    sigmar = sigmal
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

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


#BESIII  D0->pi0 e e
    #Experiment: BESIII
    #arXiv: 1802.09752
def besiii_D0topi0ee(x):
    citations.register_inspire('BESIII:2018hqu')
    q2min = [0, 1.053] #Dielectron mass
    q2max = [0.935, 1.730] #Dielectron mass
    value = [0.4e-5]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#BaBar D+->pi+ e e
    #Experiment: BaBar
    #arXiv: 1107.4465
def babar_Dptopipee(x):
    citations.register_inspire('BaBar:2011ouc')
    q2min = [0.200, 1.050] #Dielectron mass
    q2max = [0.950, 1.730] #Dielectron mass
    value = [3.9e-4]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#BaBar Ds+->K+ e e
    #Experiment: BaBar
    #arXiv: 1107.4465
def babar_DsptoKpee(x):
    citations.register_inspire('BaBar:2011ouc')
    q2min = [0.200, 1.050] #Dielectron mass
    q2max = [0.950, 1.475] #Dielectron mass
    value = [1.6e-4]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars




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

#LHCb D+->pi+ mu mu
    #Experiment: LHCb
    #arXiv: 1304.6365
def lhcb_Dptopipmumu(x):
    citations.register_inspire('LHCb:2013hxr')
    q2min = [0.250, 1.250] #Dimuonon mass
    q2max = [0.525, 2.000] #Dimuon mass
    value = [2.0e-8, 2.6e-8]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

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

#BaBar D+->K+ mu mu
    #Experiment: BaBar
    #arXiv: 1107.4465
def babar_Dptokpmumu(x):
    citations.register_inspire('BaBar:2011ouc')
    q2min = [0.200] #Dimuonon mass
    q2max = [1.475] #Dimuon mass
    value = [9.1e-4]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars


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

lhcb_KStomumu = MeasurementConstant(
    'LHCb:2020ycd',
    'flat',
    0.9e-10,
    0.6e-10,
    0.7e-10,
    max_ma=np.inf
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
def babar_Y_hadrons(x):
    citations.register_inspire('BaBar:2011kau')
    data_file_path = os.path.join(current_dir, visible, 'Babar_BR_hadrons_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars

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
def babar_Y1s_mumu(x):
    citations.register_inspire('BaBar:2012wey')
    data_file_path = os.path.join(current_dir, visible, 'Babar_BR_mumu_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars

#BaBar Y(1S)--> c c
    #Experiment: BaBar
    #arXiv:1502.06019
    #Results at 90% confidence level
    #Branching ratio
def babar_Y1s_cc(x):
    citations.register_inspire('BaBar:2015cce')
    data_file_path = os.path.join(current_dir, visible, 'Babar_BR_cc_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars

#Belle Y(1S)--> Leptons
    #Experiment: Belle
    #arXiv:2112.11852
    #Results at 90% confidence level
    #Branching ratio
def belle_Y1S_mumu(x):
    citations.register_inspire('Belle:2021rcl')
    data_file_path = os.path.join(current_dir, visible, 'Belle_BR_mumu_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars

def belle_Y1S_tautau(x):
    citations.register_inspire('Belle:2021rcl')
    data_file_path = os.path.join(current_dir, visible, 'Belle_BR_tautau_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars


#BESIII J/psi--> mu mu
    #Experiment: BESIII
    #arXiv:2109.12625
    #Results at 90% confidence level
    #Branching ratio
def besiii_Jpsi_mumu(x):
    citations.register_inspire('BESIII:2021ges')
    data_file_path = os.path.join(current_dir, visible, 'BES_BR_mumu_binned.txt')
    q2min, q2max, param = data_reading(data_file_path)
    value = []
    sigmar = []
    for ii in range(len(q2min)):
        value.append(0)
        sigmar.append(sigma(0.9, 1, param[ii]))
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmar, sigmar)
    return values, sigmals, sigmars

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

belle_B0toK0stautau = MeasurementConstantBound('Belle:2021ecr', 'prompt', 3.1e-3, conf_level=0.9, min_ma=2*mtau, lab_boost=0.28, mass_parent=mB0, mass_sibling=mKst0, rmin=100)

#belle_Y1S_tautau = MeasurementInterpolatedBound('Belle:2021rcl', os.path.join(current_dir, visible, 'Belle_BR_tautau_binned.txt'), 'prompt', conf_level=0.9, min_ma=2*mtau, lab_boost=0.42, mass_parent=mUpsilon1S, mass_sibling=0, rmin=100)
babar_Y3S_tautau = MeasurementInterpolatedBound('BaBar:2009oxm', os.path.join(current_dir, visible, 'babar_Y3S_tautau.txt'), 'prompt', conf_level=0.9, lab_boost=0.469/(1-0.469**2)**0.5, mass_parent=mUpsilon3S, mass_sibling=0, rmin=10)

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

def get_measurements(transition: str, exclude_projections: bool = True) -> dict[str, MeasurementBase]:
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

    initial, final = parse(transition)
    #Initial state B+
    if initial == ['B+'] and final == sorted(['K+', 'alp']):
        return {'Belle II': belleII_bptoknunu_lightmediator}
    elif initial == ['B+'] and final == sorted(['pion+', 'alp']):
        return {'Belle': belle_Bchargedtopichargednunu}
    elif initial == ['B+'] and final == sorted(['rho+', 'alp']):
        return {'Belle': belle_Bchargedtorhochargednunu}
    elif initial == ['B+'] and final == sorted(['K+', 'electron', 'electron']):
        return {'Belle II': belleII_bkee_displvertex}
    elif initial == ['B+'] and final == sorted(['K+', 'muon', 'muon']):
        if exclude_projections:
            return {'LHCb': lhcb_bkmumu_displvertex, 'Belle II': belleII_bkmumu_displvertex, 'CHARM': charm_bkmumu_displvertex}
        else:
            return {'LHCb': lhcb_bkmumu_displvertex, 'Belle II': belleII_bkmumu_displvertex, 'CHARM': charm_bkmumu_displvertex, 'NA62': na62proj_bkmumu_displvertex, 'SHiP': shipproj_bkmumu_displvertex}  
    elif initial == ['B+'] and final == sorted(['K+', 'tau', 'tau']):
        return {'BaBar': babar_bktautau}
    elif initial == ['B+'] and final == sorted(['K+', 'photon', 'photon']):
        return {'BaBar': babar_bkphotons_displvertex}
    elif initial == ['B+'] and final == sorted(['K+', 'pion+', 'pion-', 'pion0']):
        return {'Belle': belle_bpKomega3pi}
    elif initial == ['B+'] and final == sorted(['K+', 'eta', 'pion+', 'pion-']):
        return {'BaBar': babar_BKetapipi}
    #Initial state B0
    elif initial == ['B0'] and final == sorted(['pion0', 'alp']):
        return {'Belle': belle_B0topi0nunu}
    elif initial == ['B0'] and final == sorted(['K0', 'alp']):
        return {'Belle': belle_B0toK0nunu}
    elif initial == ['B0'] and final == sorted(['rho0', 'alp']):
        return {'Belle': belle_B0torho0nunu}
    elif initial == ['B0'] and final == sorted(['K*0', 'alp']):
        return {'BaBar': babar_btoksnunu_lightmediator}
    elif initial == ['B0'] and final == sorted(['K*0', 'electron', 'electron']):
        return {'Belle II': belleII_bks0ee_displvertex}
    elif initial == ['B0'] and final == sorted(['K*0', 'muon', 'muon']):
        return {'LHCb': lhcb_bks0mumu_displvertex, 'Belle II': belleII_bks0mumu_displvertex}
    elif initial == ['B0'] and final == sorted(['K*0', 'tau', 'tau']):
        return {'Belle': belle_B0toK0stautau}
    elif initial == ['B0'] and final == sorted(['K0', 'pion+', 'pion-', 'pion0']):
        return {'Belle': belle_b0Komega3pi}
    elif initial == ['B0'] and final == ['electron', 'electron']:
        return {'LHCb': lhcb_B0toee}
    elif initial == ['B0'] and final == ['muon', 'muon']:
        return {'LHCb': lhcb_B0tomumu, 'CMS': cms_B0tomumu}
    elif initial == ['B0'] and final == sorted(['tau', 'tau']):
        return {'LHCb': lhcb_B0totautau}
    #Initial state Bs
    elif initial == ['Bs'] and final == sorted(['phi', 'alp']):
        return {'DELPHI': delphi_Bstophinunu}
    elif initial == ['Bs'] and final == ['electron', 'electron']:
        return {'LHCb': lhcb_Bstoee}
    elif initial == ['Bs'] and final == sorted(['muon', 'muon']):
        return {'LHCb': lhcb_Bstomumu, 'CMS': cms_Bstomumu}
    elif initial == ['Bs'] and final == sorted(['tau', 'tau']):
        return {'LHCb': lhcb_Bstotautau}
    #Initial state Upsilon(3S)
    elif initial == ['Upsilon(3S)'] and final == sorted(['photon', 'tau', 'tau']):
        return {'BaBar': babar_Y3S_tautau}
    #Initial state K+
    elif initial == ['K+'] and final == sorted(['pion+', 'alp']):
        return {'NA62': na62_Ktopiinv}    
    #elif initial == ['K+'] and final == sorted(['pion+', 'photon', 'photon']):
    #    return {'NA62+NA48/2': na62na48_kpigammagamma}
    elif initial == ['K+'] and final == sorted(['muon', 'muon', 'pion+']):
        return {'NA48/2': na482_Kpimumu}
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
    #Initial state KS
    elif initial == ['KS'] and final == ['electron', 'electron']:
        return {'KLOE': kloe_KStoee}
    elif initial == ['KS'] and final == ['muon', 'muon']:
        return {'LHCb': lhcb_KStomumu}
    else:
        raise KeyError(f"No measurements for {transition}")
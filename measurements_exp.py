#File with all possible experimental measurement
import os
import numpy as np
from scipy.stats import chi2
from ...citations import citations
# Get the directory of the current script
current_dir = os.path.dirname(__file__)

#Meson mass (useful)
mB = 5.279 #GeV
mB0 = 5.279 #GeV
mBs = 5.366 #GeV
mK = 0.493 #GeV
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
        if x>qmin[ii] and x<qmax[ii]:
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
#BELLEII B+->K+ nu nu 2023
    #Experiment: BELLE II
    #arXiv: 2311.14647
    #Branching ratio
def belleII_BtoKnunu(x):
    #INPUT:
        #x:
    citations.register_inspire('Belle-II:2023esi') 
    q2min = [0]
    q2max = [mB0**2]
    value = [2.3e-5]
    sigmal = [0.7e-5]
    sigmar = [0.7e-5]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

#NA62 K+->pi+ nu nu 2021
    #Experiment: NA62
    #arXiv: 2103.15389
    #Branching ratio
def na62_Ktopinunu(x):
    citations.register_inspire('NA62:2021zjw') 
    q2min = [0]
    q2max = [mK**2]
    value = [10.6e-11]
    sigmal = [4.1e-11]
    sigmar = [3.5e-11]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

#NA62 K+->pi+ pi0(->X) 2020 
    #Experiment: NA62
    #arXiv: 2010.07644
    #Branching ratio
def na62_pi0toinv(x):
    citations.register_inspire('NA62:2020pwi') 
    q2min = [0.110]
    q2max = [0.155]
    value = [4.4e-9]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#J-PARC KOTO KL->pi0 nu nu
    #Experiment: KOTO
    #arXiv: 1810.09655
    #Branching ratio
def koto_kltopi0nunu(x):
    citations.register_inspire('KOTO:2018dsc') 
    q2min = [0]
    q2max = [0.261]
    value = [3.0e-9]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#J-PARC KOTO KL->pi0 inv
    #Experiment: KOTO
    #arXiv: 1810.09655
    #Branching ratio
def koto_kltopi0inv(x):
    citations.register_inspire('KOTO:2018dsc') 
    q2min = [0]
    q2max = [0.261]
    value = [2.4e-9]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar B+->K+ nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_bptokpnunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [3.7e-5]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar B+->K*+ nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_bptokstarpnunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [11.6e-5]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar B0->K0 nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_b0tok0nunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [8.1e-5]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar B0->K*0 nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio
def babar_b0tokstar0nunu(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [9.3e-5]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar B->K nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_btoknunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [3.2e-5]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar B->K* nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_btokstarnunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [7.9e-5]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar J/psi-> nu nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_jpsitonunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [3.9e-3]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

#BaBar psi(2S)-> nu
    #Experiment: BaBar
    #arXiv: 1303.7465
    #Branching ratio, combined
def babar_psi2stonunu_comb(x):
    citations.register_inspire('BaBar:2013npw') 
    q2min = [0]
    q2max = [4.785]
    value = [15.5e-3]
    sigma = sigma(0.9, 1, value)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars


############ Quarkonia decays ############
invisible = "Binned_measurement/Invisible/"

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
def belle_BchargedtoKchargednunu(x):
    citations.register_inspire('Belle:2017oht')
    q2min = [0]
    q2max = [mB**2]
    value = [4e-5]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

def belle_Bchargedtorhochargednunu(x):
    citations.register_inspire('Belle:2017oht')
    q2min = [0]
    q2max = [mB**2]
    value = [3e-5]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

def belle_Bchargedtopichargednunu(x):
    citations.register_inspire('Belle:2017oht')
    q2min = [0]
    q2max = [mB**2]
    value = [1.4e-5]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

def belle_B0toK0nunu(x):
    citations.register_inspire('Belle:2017oht')
    q2min = [0]
    q2max = [mB0**2]
    value = [2.6e-5]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

def belle_B0toK0starnunu(x):
    citations.register_inspire('Belle:2017oht')
    q2min = [0]
    q2max = [mB0**2]
    value = [1.8e-5]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

def belle_B0topi0nunu(x):
    citations.register_inspire('Belle:2017oht')
    q2min = [0]
    q2max = [mB0**2]
    value = [9e-6]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

def belle_B0torho0nunu(x):
    citations.register_inspire('Belle:2017oht')
    q2min = [0]
    q2max = [mB0**2]
    value = [4e-5]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0] #Estimated value
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars

#BESIII D0->pi0 nu nu 2021 
    #Experiment: BESIII
    #arXiv: 2112.14236
    #@ 90% confidence level
    #Branching ratio
def besIII_D0topi0nunu(x):
    citations.register_inspire('BESIII:2021slf')
    q2min = [0]
    q2max = [mD0**2]
    value = [2.1e-4]
    cl = 0.9
    df = 1 
    sigmas = sigma(cl, df, value)
    valuep = [0]
    #sigmal = [4.1e-11]
    #sigmar = [3.5e-11]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, valuep, sigmas, sigmas)
    return values, sigmals, sigmars



#BelleII e+e- -> gamma a
    #Experiment: BelleII
    #arXiv: 2007.13071
    #@90% confidence level
    #Cross section
def belleII_upsilon4s(x):
    citations.register_inspire('Belle-II:2020jti')
    data_file_path = os.path.join(current_dir, invisible, 'Belle2_gamma_binned.txt')
    q2min, q2max, value = data_reading(data_file_path)
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, 0.5*np.array(value), 0.5*np.array(value))
    return values, sigmals, sigmars

#################################### VISIBLE SEARCHES ####################################

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
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars

#NA48/2 K-> pi e e
    #Experiment: NA48/2
    #arXiv: 0903.3130
    #Branching ratio
def na48_Ktopiee(x):
    citations.register_inspire('NA482:2009pfe')
    q2min = [0]
    q2max = [mK**2]
    value = [3.11e-7]
    sigmal = [0.12e-7]
    sigmar = [0.12e-7]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmar)
    return values, sigmals, sigmars


#LHCb Bs->mu mu
    #Experiment: LHCb
    #arXiv: 1703.05747
    #Branching ratio
def lhcb_Bstomumu(x):
    citations.register_inspire('LHCb:2017rmj')
    q2min = [0]
    q2max = [mBs**2]
    value = [3.0e-9]
    sigmal = [np.sqrt((0.6)**2+(0.3)**2)*1e-9]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmal)
    return values, sigmals, sigmars


#PDG Kl-> mu mu
    #Experiment: PDG
    #DOI: 10.1103/PhysRevD.110.030001
    #Branching ratio
def pdg_Kltomumu(x):
    citations.register_inspire('ParticleDataGroup:2024cfk')
    q2min = [0]
    q2max = [mKl**2]
    value = [6.84e-9]
    sigma = [0.11e-9]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigma, sigma)
    return values, sigmals, sigmars

### Decays to gamma gamma

#NA62  K+->pi+ gamma gamma
    #Experiment: NA62
    #arXiv: 1402.4334
def na62_Ktopigammagamma(x):
    citations.register_inspire('NA62:2014ybm')
    q2min = [0.220**2] #Digamma momentum
    q2max = [0.354**2] #Digamma momentum
    value = [9.65e-7]
    sigmal = [0.63e-7]
    sigmar = sigmal
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmas, sigmas)
    return values, sigmals, sigmars

#NA48  KL->pi0 gamma gamma
    #Experiment: NA48
    #arXiv: hep-ex/0205010
def na48_Kltopi0gammagamma(x):
    citations.register_inspire('NA48:2002xke')
    q2min = [0] #Digamma momentum
    q2max = [0.110**2] #Digamma momentum
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
    citations.register_inspire('')
    q2min = [0] #Digamma momentum
    q2max = [0.363**2] #Digamma momentum
    value = [1.29e-6]
    sigmas = [np.sqrt(0.03**2+0.05**2)*1e-6]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmas, sigmas)
    return values, sigmals, sigmars


#E949  K+->pi+ gamma gamma
    #Experiment: E949
    #arXiv: hep-ex/0505069
def E949_Ktopigammagamma(x):
    citations.register_inspire('E949:2005qiy')
    q2min = [0.213**2] #Pion momentum
    q2max = [0.227**2] #Pion momentum
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
def E787_Ktopigammagamma(x):
    citations.register_inspire('E787:1997abk')
    q2min = [0.100**2] #Pion momentum
    q2max = [0.180**2] #Pion momentum
    value = [6.0e-7]
    sigmal = [np.sqrt((1.5)**2+(0.7)**2)*1e-7]
    values, sigmals, sigmars = bin_selection(x, q2min, q2max, value, sigmal, sigmal)
    return values, sigmals, sigmars



############ Quarkonia decays ############
visible = "Binned_measurement/Visible/"

#BaBar Y(2S, 3S)--> Hadrons
    #Experiment: BaBar
    #arXiv:1108.3549
    #Results at 90% confidence level
    #Branching ratio
def babar_Jpsi_hadrons(x):
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

#BaBar Y(1S)--> Muons
    #Experiment: BaBar
    #arXiv:1210.0287
    #Results at 90% confidence level
    #Branching ratio
def babar_Jpsi_mumu(x):
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
def babar_Jpsi_cc(x):
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
def belle_Y1S_mumu(x):
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

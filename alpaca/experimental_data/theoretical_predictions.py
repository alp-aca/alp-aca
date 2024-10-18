#File with all possible theoretical predictions
from ..citations import citations

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


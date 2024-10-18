import numpy as np
import flavio
import vegas as vegas
import functools
from . import threebody_decay


#IMPORTANT NOTE:
#Simplification in data-driven function interpolation

#ALP decays to different channels (leptonic, hadronic, photons)

mpi0 = 134.97e-3 #Mass of pi0 (in GeV)
mpi = mpi0
meta = 547.86e-3 #Mass of eta (in GeV)
metaprime = 957.78e-3 #Mass of eta' (in GeV)
#deltaI = 1/3 #(mu-md)/(mu+md) Parametrises isospin breaking
sw2 = 0.23 #Weak mixing angle
mW = 80.377 #Mass of W (in GeV) 
mK = 493.677e-3 #GeV, Mass K meson
mrho = 775.26e-3 #GeV, Mass rho meson
fpi = 130e-3 #GeV, decay rate Pion


#Decay widths (ESTIMATIONS)
Gammasigma = 400e-3 #Gamma sigma (in GeV)
Gammaf0 = 55e-3 #Gamma f0 (in GeV)
Gammaa0 = 75e-3 #Gamma a0 (in GeV)
Gammaf2 = 186.6e-3 #Gamma f2 (in GeV)
Gammarho = 150e-3 #Gamma rho (in GeV)


#NEED TO INCLUDE KINEMATIC RELATIONS
me = 0.5109989500e-3 #GeV, mass electron
mmu = 105.6583755e-3 #GeV, mass muon
mtau= 1.77693 #GeV, mass tau
mtop = 174 #GeV, top mass
mcharm=1.27#GeV, charm mass
mup = 2.16e-3 #GeV, up mass

mbottom = 4.183 #GeV, top mass
mstrange=93.5e-3#GeV, charm mass
mdown = 4.70e-3 #GeV, up mass

mlepton=[me,mmu,mtau]
mqup=[mup,mcharm,mtop]
mqdown=[mdown,mstrange,mbottom]
mquark=[mqup,mqdown]
clepton=[0,0,0]
cquark=[clepton,clepton]
fav=1000

pars = flavio.default_parameters.get_central_all()
alphaem = lambda q: flavio.physics.running.running.get_alpha_e(pars, q)
alphas = lambda q: flavio.physics.running.running.get_alpha_s(pars, q)
g=np.sqrt(12*np.pi)#0.6 

mu = mup
md = mdown
ms = mstrange


#Coupling matrix        Mass matrix
#   cu  cc  ctop           mu   mc  mtop 
# C=                    M=
#   cd  cs  cbot           md   ms  mbot

######################################################   PHOTON CHANNEL (a-> gamma gamma)    ######################################################

#Chiral contribution (1811.03474, eq. S26, ,approx) (below mass eta')
def cgammachiral(ma):
    #INPUT:
        #ma: Mass of ALP (GeV)
    #OUTPUT:
        #Contribution from chiral transformation
    if ma <= metaprime:
        cgammachi = 1
    else: cgammachi = 0
    return cgammachi

#Vector meson dominance (1811.03474, eq. S27) (below 2.1 GeV)
def ffunction(ma):
    #INPUT:
        #ma: Mass of ALP (GeV)
    #OUTPUT:
        #Data-driven function 
    if ma < 1.4: fun = 1
    elif ma >= 1.4 and ma <= 2: 
        fun = ((1.4/2)**4-1)/(2-1.4)*(ma-1.4) + 1 #Interpolation (for now I do just straight line) (y2-y1)/(x2-x1)*(x-x1)+y1
    else: fun = (1.4/ma)**4
    return fun

def alphasbar(ma):
    #INPUT:
        #ma: Mass of ALP (GeV)
    #OUTPUT:
        #alpha_strong
    if ma <= 1:
        astrong = 1
    else: astrong = alphas(ma)
    return astrong

def cgammaVMD(ma, model):
    #INPUT:
        #ma: Mass of ALP (GeV)
        #model: Coupling of the model presented
    #OUTPUT:
        #Contribution from vector meson dominance
    deltaI = 0
    if ma < 2.1 :
            #aux = alpVV(ma, model, deltaI)
            #cgvmd = -ffunction(ma)*(3*aux[0]+2*aux[1]+1/3*aux[2]+2/3*aux[3])# 2*alphasbar(ma)/(3*np.sqrt(6))* (4*coup[0]+ coup[1]+ coup[2])
            aux = calpmixing(ma, model, deltaI)
            cgvmd = -ffunction(ma)*2*alphasbar(ma)/(3*np.sqrt(6))*(4*aux[0] + aux[1] + aux[2])# 2*alphasbar(ma)/(3*np.sqrt(6))* (4*coup[0]+ coup[1]+ coup[2])
    else: cgvmd = 0
    return cgvmd

#pQCD-based contribution (from 1708.00443, Eq. 13, 14, 16)
#Loop functions
def ftau(xvar):
    #INPUT:
        #xvar: Dimensionless variable
    #OUTPUT:
        #Loop function (as defined in Eq. 14, 1708.00443)
    if xvar<1:
        fres= np.pi/2+1j/2*np.log((1+np.sqrt(1-xvar))/(1-np.sqrt(1-xvar)))
    else:
        fres=np.arcsin(1/np.sqrt(xvar))
    return fres

def loopB1(ma, m):
    #INPUT:
        #ma: Mass of ALP (in GeV)
        #m: Mass of SM particle (in GeV)
    #OUTPUT:
        #Loop function B1 (as defined in Eq. 14, 1708.00443)
    xvar = 4*pow(m/ma,2)
    return 1-xvar*pow(ftau(xvar),2)

def loopB2(ma, m):
    #INPUT:
        #ma: Mass of ALP (in GeV)
        #m: Mass of particle (in GeV)
    #OUTPUT:
        #Loop function B2 (as defined in Eq. 14, 1708.00443)
    xvar = 4*pow(m/ma,2)
    return 1-(xvar-1)*pow(ftau(xvar),2)

def cgammapQCD(ma, ml, mq, cl, cq, cFF, cWW, cGG,fa):
    #INPUT:
        #ma: Mass of ALP (GeV)
        #ml: Array with lepton masses (me, mmu, mtau)
        #mq: Array with quark masses
        #cl: Array with lepton couplings (ce, cmu, ctau)
        #cq: Array with quark couplings
        #cFF: Tree level coupling a-gamma-gamma
        #cWW: Tree level coupling a-W-W
        #cGG: Tree level coupling a-G-G
        #fa: Scale of ALP (in GeV)
    #OUTPUT:
        #Contribution from pQCD
    
    if ma> 1.5:
        clep =(cl[0]*loopB1(ma, ml[0])+cl[1]*loopB1(ma,ml[1])+cl[2]*loopB1(ma,ml[2]))
        cquarku = (cq[0][0]*loopB1(ma,mq[0][0])+cq[0][1]*loopB1(ma,mq[0][1])+cq[0][2]*loopB1(ma,mq[0][2]))
        cquarkd = (cq[1][0]*loopB1(ma,mq[1][0])+cq[1][1]*loopB1(ma,mq[1][1])+cq[1][2]*loopB1(ma,mq[1][2]))
        cferm =  (-1)**2*clep + 3*(2/3)**2*cquarku + 3*(-1/3)**2*cquarkd #Eq.13
        Lambda = -32*np.pi**2*fa*cGG
        cglu2loop = (2/3)**2*(loopB1(ma,mq[0][0])*np.log(Lambda**2/mpi0**2)+loopB1(ma,mq[0][1])*np.log(Lambda**2/mq[0][1]**2)+loopB1(ma,mq[0][2])*np.log(Lambda**2/mq[0][2]**2))+\
        (-1/3)**2*(loopB1(ma,mq[1][0])*np.log(Lambda**2/mpi0**2)+loopB1(ma,mq[1][1])*2*np.log(Lambda**2/mpi0**2)+loopB1(ma,mq[1][2])*2*np.log(Lambda**2/mq[1][2]**2))    #Eq.16
        cpQCD = cFF + 1/(16*np.pi**2)*cferm + 2*alphaem(ma)/np.pi* cWW/sw2*loopB2(ma,mW) -(3*alphas(ma)**2/np.pi**2*cGG*cglu2loop)
    #Lambda = 32*np.pi**2*fa*cGG
    #if ma > 2.1: cpQCD=alphas(ma)**2/(6*np.pi**2)*2*(5*np.log(Lambda/mpi0)+np.log(Lambda/mK))-alphas(ma)**2*ma**2/(72*np.pi**2)*2*(4*np.sqrt(3)/mq[0][1]**2*np.log(Lambda/mq[0][1])+1/mq[1][2]**2*np.log(Lambda/mq[1][2])+4/mq[0][2]**2*np.log(Lambda/mq[1][2]))
    #elif ma>1.6 and ma<2.1: cpQCD=-alphas(ma)**2*ma**2/(72*np.pi**2)*2*(4*np.sqrt(3)/mq[0][1]**2*np.log(Lambda/mq[0][1])+1/mq[1][2]**2*np.log(Lambda/mq[1][2])+4/mq[0][2]**2*np.log(Lambda/mq[1][2]))
    else: cpQCD =0 
    return cpQCD

#Decay rate a-> gamma gamma
def atogammagamma(ma, ml, mq, model, cl, cq, cFF, cWW, cGG, fa):
    #INPUT:
        #ma: Mass of ALP (GeV)
        #ml: Array with lepton masses (me, mmu, mtau)
        #mq: Array with quark masses 
        #cl: Array with lepton couplings (ce, cmu, ctau)
        #cq: Array with quark couplings
        #cFF: Tree level coupling a-gamma-gamma
        #cWW: Tree level coupling a-W-W
        #cGG: Tree level coupling a-G-G
        #fa: Scale of ALP (in GeV)
    #OUTPUT:
        #Decay rate (in GeV)
    cont = cgammachiral(ma) + cgammaVMD(ma, model) + cgammapQCD(ma, ml, mq, cl, cq, cFF, cWW, cGG, fa)
    #cont = cgammapQCD(ma, ml, mqu, mqd, cl, cqu, cqd, cFF, cWW, cGG, fa)
    return alphaem(ma)**2*ma**3/(pow(4*np.pi,3)*fa**2)*cont*np.conjugate(cont)

######################################################   HADRONIC CHANNELS    ######################################################
#Extracted from 1811.03474
#Following U(3) representation of ALPs 
# a=alphabar_S(ma)/sqrt(6)*diag(Cu,Cd,Cs)
# alphabar_S(ma)= 1 (if ma<=1 GeV), alpha_S(ma) (if ma>1 GeV)



###########################    ALP mixing   ###########################

def u3rep(ma, coup, deltaI):
    #Input
        #ma: Mass of the ALP (GeV)
        #coup: Vector of couplings (cu, cd, cs, cg)
        #coup[3]: Coupling to gluons
        #deltaI: (md-mu)/(md+mu)
    #Return
        #api0: Mixing ALP-pi0
        #aeta: Mixing ALP-eta
        #aetaprime: Mixing ALP-etaprime
    
    #Auxiliary functions (needed for easiness)
        #Elements of the kinetic mixing matrix
    Kapi = (coup[0]-coup[1])/(128*coup[3]*np.pi**2)+ 1/2*ms*(md-mu)/G(mu,md,ms)
    Kaeta = (coup[0]+coup[1]-coup[2])/(64*np.sqrt(6)*coup[3]*np.pi**2)+1/np.sqrt(6)-2*md*mu/(np.sqrt(6)*G(mu,md,ms))
    Kaetap = (coup[0]+coup[1]+2*coup[2])/(256*np.sqrt(3)*coup[3]*np.pi**2)+1/(4*np.sqrt(3))+mu*md/(4*np.sqrt(3)*G(mu,md,ms))

    api0 = 1/(ma**2 - mpi**2)*(Mapi2 +ma**2 * Kapi + \
                               deltaI*(Mpieta2*(Maeta2 + ma**2 *Kaeta)/(ma**2 - meta**2) + Mpietap2 *(Maetap2 + ma**2* Kaetap)/(ma**2 - metaprime**2)))
    
    
    aeta = 1/(ma**2 - meta**2)* (Maeta2 +ma**2 *Kaeta +\
                                 deltaI *(Mpieta2* (Mapi2 + ma**2*Kapi)/(ma**2 - mpi**2)+ Metaetap2 *(Maetap2 + ma**2 *Kaetap)/(ma**2 - metaprime**2)))
    
    
    aetaprime = 1/(ma**2-metaprime**2)* (Maetap2+ ma**2*Kaetap +\
                                        deltaI *(Mpietap2 *(Mapi2 + ma**2*Kapi)/(ma**2 - mpi**2) + Metaetap2* (Maeta2 + ma**2* Kaeta)/(ma**2 - meta**2)))
    
    return api0, aeta, aetaprime

def calpmixing(ma, coup, deltaI):
    #INPUT:
        #ma: Mass of the ALP (GeV)
        #coup: Vector of couplings (cu, cd, cs, cg)
    #OUTPUT:
        #Cu,Cd,Cd: Mixing

    #Pseudoscalar matrix generators
    pi0gen = 1/2*np.array([1, -1, 0])
    etagen =1/np.sqrt(6)*np.array([1, 1, -1])
    etapgen =1/(2*np.sqrt(3))*np.array([1, 1, 2])
    #Mixing elements
    api0, aeta, aetap = u3rep(ma, coup, deltaI)
    #C mixing
    Cu = np.sqrt(6)*(api0*pi0gen[0] + aeta*etagen[0] + aetap*etapgen[0])/alphasbar(ma)
    Cd = np.sqrt(6)*(api0*pi0gen[1] + aeta*etagen[1] + aetap*etapgen[1])/alphasbar(ma)
    Cs = np.sqrt(6)*(api0*pi0gen[2] + aeta*etagen[2] + aetap*etapgen[2])/alphasbar(ma)
    return Cu, Cd, Cs

def alpVV(ma, c, deltaI):
    #INPUT:
        #ma: Mass of the ALP (GeV)
        #c: Vector of couplings (cu, cd, cs, cg)
    #OUTPUT:
        #<a rho rho>: Mixing element
    arhorho = (np.sqrt(2)* u3rep(ma, c, deltaI)[1] + u3rep(ma, c, deltaI)[2])/(2* np.sqrt(3))
    arhow = u3rep(ma, c, deltaI)[0]/2
    aww = (np.sqrt(2)* u3rep(ma, c, deltaI)[1] + u3rep(ma, c, deltaI)[2])/(2* np.sqrt(3))
    aphiphi = -u3rep(ma,c,deltaI)[1]/np.sqrt(6) + u3rep(ma,c,deltaI)[2]/np.sqrt(3)
    return arhorho, arhow, aww, aphiphi

def G(x, y, z):
    return x*y + x*z + y*z

#ALP-meson mixing
    #Elements of the mass mixing matrix

Mpipi2 = mpi**2 
Mpieta2 = - Mpipi2*np.sqrt(2/3)
Mpietap2 = - Mpipi2/np.sqrt(3)
Mapi2 = 0
Maeta2 = -np.sqrt(2/3)*mpi/(mu+md)*mu*md*ms/G(mu,md,ms)
Maetap2 = Maeta2*2*np.sqrt(2)
Metaetap2 = 0


#Spp coefficients
def Spp(mP,mPp,MPPp2): #Eq. S8
    return MPPp2/(mP**2-mPp**2)

Setapi0 = Spp(meta,mpi0,Mpieta2)
Setappi0 = Spp(metaprime,mpi0,Mpietap2)
Setapeta = Spp(metaprime,meta,Metaetap2)



###########################    BREIT-WIGNER DISTRIBUTION   ###########################
#Breit-Wigner expression (from Eqs. 27-33 of PHYSICAL REVIEW D 86, 032013 (2012), arXiv:1205.2228)
def beta(x):
    #INPUT:
        #x: Energy^2 (in GeV)
    #OUTPUT:
        #beta function
    return np.sqrt(1-4*mpi**2/x)

def gammaf(s, m, Gamma):
    #INPUT
        #s: CM energy (in GeV^2)
        #m: Mass of unstable propagating particle (in GeV)
        #Gamma: Decay width of unstable propagating particle (in GeV)
    #OUTPUT
        #result: Modified decay width (in GeV)
    result = Gamma* (s/m**2)* (beta(s)/beta(m**2))**3
    return result

def d(m):
    #INPUT:
        #m: Mass of unstable propagating particle (in GeV)
    #OUTPUT:
        #d: Auxiliary function in GS model
    kaux = faux(m**2)[0]
    return 3/np.pi* mpi**2/kaux**2* np.log((m+2*kaux)/(2*mpi))+ m/(2*np.pi*kaux)- mpi**2*m/(np.pi*kaux**3)

def faux(x):
    #INPUT:
        #x: CM energy (in GeV^2)
    #OUTPUT:
        #kam: Auxiliary function 1
        #ham: Auxiliary function 2
        #hpam: Derivative of auxiliary function 2
    kam = 1/2*np.sqrt(x)*beta(x)
    ham = 2/np.pi * kam/np.sqrt(x)* np.log((np.sqrt(x)+2*kam)/(2*mpi))
    hpam = (-4*mpi**2+x+np.sqrt(x*(x-4*mpi**2))+4*mpi**2*(1+np.sqrt(1-4*mpi**2/x))*np.log((np.sqrt(x)+np.sqrt(x-4*mpi**2))/(2*mpi)))/\
        (2*np.pi*x* (-4*mpi**2+ x+ np.sqrt(x*(x-4*mpi**2)))) #Obtained deriving with Mathematica
    return kam, ham, hpam

def f(s, m, Gamma):
    #INPUT:
        #s: CM energy (in GeV^2)
        #m: Mass of unstable propagating particle (in GeV)
        #Gamma: Decay width of unstable particle (in GeV^2)
    #OUTPUT:
        #f: Auxiliary function in GS model
    kaux = faux(m**2)
    kaux2  = faux(s)
    return Gamma*m**2/(kaux[0]**3)* ((kaux2[0]**2)*(kaux2[1]-kaux[1])+(m**2-s)*kaux[0]**2*kaux[2])

def bw(s, m, Gamma, c):
    #INPUT
        #s: CM energy (in GeV^2)
        #m: Mass of unstable propagating particle (in GeV)
        #Gamma: Decay width of unstable propagating particle (in GeV)
        #c: Control digit (c = 1 for rho, rho', rho'', rho''', c = 0 for the rest)
    #OUTPUT
        #result: Breit-Wigner modified propagator
    if c==0:
        result= m**2/(m**2 -s -1.j*m*Gamma)
    elif c==1:
        result = m**2*(1+d(m)*Gamma/m)/(m**2-s+f(s, m, Gamma)-1.j*m*gammaf(s,m,Gamma))
    else: 
        print('Wrong control digit in BW function')
        result = 0
    return result

###########################    FUNCTION F (as defined in Eq.14)    ###########################

def ffunction(ma):
    #INPUT:
        #ma: Mass of ALP (GeV)
    #OUTPUT:
        #Data-driven function 
    if ma < 1.4: fun = 1
    elif ma >= 1.4 and ma <= 2: 
        fun = ((1.4/2)**4-1)/(2-1.4)*(ma-1.4) + 1 #Interpolation (for now I do just straight line) (y2-y1)/(x2-x1)*(x-x1)+y1
    else: fun = (1.4/ma)**4
    return fun

###########################    DECAY TO 3 PIONS a-> pi pi pi    ###########################
#Decay to 3 neutral pions 3 pi0
def ampato3pi0(ma, m1, m2, m3, model, x, kinematics): #Eq. S31
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV)
        #model: Model coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a->3 pi0 (without prefactor)
    deltaI = 1/3
    coef = u3rep(ma, model, deltaI)
    aux = coef[0] - deltaI*(1/np.sqrt(3)+np.sqrt(2)*Setapi0+Setappi0)*(np.sqrt(2)*coef[1]+coef[2]) + \
    np.sqrt(3)*deltaI*(mpi**2-2*meta**2)/(mpi**2-4*meta**2)*(np.sqrt(2)*Setapi0+Setappi0)
    return mpi**2*aux

#Decay to pi+ pi- pi0
def ampatopicharged(ma, m1, m2, m3, model, x, kinematics): #Eq.S32
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV)
        #model: Model coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a->3 pi0 (without prefactor)
    deltaI = 1/3
    coef = u3rep(ma, model, deltaI)
    mpipipm = x[0]
    aux = (3*mpipipm**2-ma**2-2*mpi**2)*coef[0] - deltaI*mpi**2*(1/np.sqrt(3)+np.sqrt(2)*Setapi0+Setappi0)*(np.sqrt(2)*coef[1]+coef[2]) + \
    deltaI*(mpi**2-2*meta**2)/(mpi**2-4*meta**2)*(np.sqrt(3)*mpi**2*(np.sqrt(2)*Setapi0+Setappi0)-3*mpipipm**2+ma**2+3*mpi**2)
    return 1/3*aux

#Decay rate (numerical integration)
k = 2.7 # Mean value of k-factor to reproduce masses of eta, eta'
def ato3pi(ma, m1, m2, m3, model, fa, c): #Eq. S33
    #INPUT:
        #ma: Mass of the ALP (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) --> Pions
        #fa: Scale of U(1)PQ (in GeV)
        #c: Control value (c=0-> Neutral pions, c=1-> pi0, pi+, pi-)
    #OUTPUT: 
        #Decay rate including symmetry factors
    if c == 0:
        s = 3*2 #Symmetry factor
        if ma > 3*mpi+0.001 and ma<=metaprime: 
            result, error = threebody_decay.decay3body(ampato3pi0, ma, m1, m2, m3, model) #Amplitude of decay to 3 neutral pions
        else: result, error = [0,0]
        #result2, error2 = threebody_decay2.decay3body(ampato3pi0, ma, m1, m2, m3) #Amplitude of decay to 3 neutral pions
    elif c == 1:
        s = 1 #Symmetry factor
        if ma > 3*mpi+0.001 and ma<=metaprime:  #mpi0+mpim+mpip
            result, error = threebody_decay.decay3body(ampatopicharged, ma, m1, m2, m3, model) #Amplitude of decay to pi+ pi- pi0
        else: result, error = [0,0] 
        #result2, error2 = threebody_decay2.decay3body(ampato3pi0, ma, m1, m2, m3) #Amplitude of decay to 3 neutral pions
    return k/(2*ma*s)*1/pow(fpi*fa,2)*result, k/(2*ma*s)*1/pow(fpi*fa,2)*error#,k/(2*ma*s)*1/pow(fpi*fa,2)*result2, k/(2*ma*s)*1/pow(fpi*fa,2)*error2


###########################    DECAY TO  a-> eta pi pi    ###########################
#It is assumed that Fpppp(m)=Fspp(m)=Ftpp(m)=F(m)

def ampatoetapipi(ma, m1, m2, m3, model, x, kinematics):
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta)
        #model: Coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a-> eta pi pi (without prefactor)
    
    #(obtained from hep-ph/9902238, Tab.II, first column)
    #xsigmapipi = 7.27    xsigmaetaeta = 3.90    xsigmaetaetap = 1.25    xsigmaetapetap = -3.82
    #xf0pipi = 1.47    xf0etaeta = 1.50    xf0etaetap = -10.19    xf0etapetap = 1.04
    #xa0pieta = -6.87    xa0pietap = -8.02

    deltaI = 0
    #Kinematic relations
    mpi1pi2 = np.sqrt(ma**2+m3**2-2*ma*x[0])
    pappi1 = kinematics[0]
    pappi2 = kinematics[1]
    papeta = kinematics[2]
    ppi1ppi2 = kinematics[3]
    petappi1 = kinematics[4]
    petappi2 = kinematics[5]
    petaqpipi = petappi1 - petappi2
    petappipi = petappi1 + petappi2
    ppipi2 = mpi1pi2**2
    qpipi2 = m1**2+m2**2-2*ppi1ppi2
    metapi1 = np.sqrt(m1**2+m3**2+2*kinematics[4])
    metapi2 = np.sqrt(m2**2+m3**2+2*kinematics[5])


    ####DECAY a->eta pi pi
    #aetasigma = 1/3*(2*u3rep(ma, model, deltaI)[2]*Setapeta*xsigmaetapetap - 6*u3rep(ma, model, deltaI)[1]*xsigmaetaeta -\
     #                3*u3rep(ma, model, deltaI)[2]*xsigmaetaetap + u3rep(ma, model, deltaI)[1]*Setapeta*xsigmaetaetap + np.sqrt(2)*u3rep(ma, model, deltaI)[0]*Setapi0*xsigmapipi)
    aetasigma = ((np.sqrt(2) + 2 *np.sqrt(10))*u3rep(ma, model, deltaI)[1]+ 2* (-1 + np.sqrt(5))* u3rep(ma, model, deltaI)[2])/(12 *np.sqrt(11))    
    #7.27/(fpi/fav* api0) #Eq.A4
    #aetaf0 = 1/3*(2*u3rep(ma, model, deltaI)[2]*Setapeta*xf0etapetap - 6*u3rep(ma, model, deltaI)[1]*xf0etaeta -\
     #             3*u3rep(ma, model, deltaI)[2]*xf0etaetap + u3rep(ma, model, deltaI)[1]*Setapeta*xf0etaetap + np.sqrt(2)*u3rep(ma, model, deltaI)[0]*Setapi0*xf0pipi)
    aetaf0 = (-2*(-1 + np.sqrt(2))*u3rep(ma, model, deltaI)[1]+ (4 + np.sqrt(2))* u3rep(ma, model, deltaI)[2])/(12 *np.sqrt(5)) 
    # 1.47/(fpi/fav* api0)
    #api0a0 = 1/3*((-3*u3rep(ma, model, deltaI)[1] + u3rep(ma, model, deltaI)[2]*Setapi0)*xa0pieta + (-3*u3rep(ma, model, deltaI)[2]+ u3rep(ma, model, deltaI)[1]*Setappi0)*xa0pietap)
    api0a0 = (np.sqrt(2)*u3rep(ma, model, deltaI)[1] + u3rep(ma, model, deltaI)[2])/(4 *np.sqrt(3))
    #-6.87/(fpi/fav* api0)
    aetaf2 = 1/12* (2* u3rep(ma, model, deltaI)[1]+ np.sqrt(2)*u3rep(ma, model, deltaI)[2])

    ####DECAY a->etaprime pi pi
    aetapsigma = (np.sqrt(2)*(-1 + np.sqrt(5))*u3rep(ma, model, deltaI)[1]+ (2 + np.sqrt(5))* u3rep(ma, model, deltaI)[2])/(6 *np.sqrt(22))    
    aetapf0 = ((4 + np.sqrt(2))*u3rep(ma, model, deltaI)[1]+ (1 - 4*np.sqrt(2))* u3rep(ma, model, deltaI)[2])/(12 *np.sqrt(5)) 
    aetapf2 = 1/12* (np.sqrt(2)* u3rep(ma, model, deltaI)[1]+ u3rep(ma, model, deltaI)[2])


    #Mix amplitude (Eq.S48)
    amix = 0 #Approximation --> (np.sqrt(2)*aeta0+aeta8)*mpi**2/(3*fpi**2)*ffunction(ma) 

    #a-> Sigma (Eq.S49)
    if mpi1pi2< 2*mK: 
        asigma = -(10)**2* aetasigma* papeta* ppi1ppi2*bw(ma**2, mpi1pi2, Gammasigma, 0)*ffunction(ma)
    else: asigma = 0 #Avoid unitarity violation

    #a-> f0 (Eq.S50)
    af0 = (7.3)**2* aetaf0* papeta* ppi1ppi2* bw(ma**2, mpi1pi2, Gammaf0, 0)*ffunction(ma) 

    #a-> a0 (Eq.S51)
    aa0 = (13)**2* api0a0* ffunction(ma)* (pappi2* petappi1*bw(ma**2, metapi1, Gammaa0, 0) + pappi1* petappi2* bw(ma**2, metapi2, Gammaa0,0)) 

    #a-> f2 (Eq.)
    af2 = (16)**2* aetaf2* (petaqpipi**2 - 1/3*qpipi2*(m3**2+ petappipi**2/ppi1ppi2**2- 2*petappipi**2/ppi1ppi2**2))*\
          bw(ma**2, mpi1pi2, Gammaf2, 0)* ffunction(ma)
    aux = (amix+asigma+af0+aa0+af2) 
    return aux

def atoetapipi(ma, m1, m2, m3, model, fa, c): #Eq. S33
    #INPUT:
        #ma: Mass of the ALP (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta)
        #fa: Scale of U(1)PQ (in GeV)
        #c: Control value (c=0-> Neutral pions, c=1-> pi0, pi+, pi-)
    #OUTPUT: 
        #Decay rate including symmetry factors
    if c == 0:
        s = 2 #Symmetry factor
        if ma > meta + 2*mpi:
            result, error = threebody_decay.decay3body(ampatoetapipi, ma, m1, m2, m3, model) #Amplitude of decay to pi+ pi- eta
        else: result, error = [0, 0]

    elif c == 1:
        s = 1 #Symmetry factor
        if ma > meta + 2*mpi:
            result, error = threebody_decay.decay3body(ampatoetapipi, ma, m1, m2, m3, model) #Amplitude of decay to pi+ pi- eta
        else: result, error = [0, 0]
    return 1/(2*ma*s)*pow(fpi/fa,2)*result, 1/(2*ma*s)*pow(fpi/fa,2)*error#, 1/(2*ma*s)*pow(fpi/fa,2)*result2, 1/(2*ma*s)*pow(fpi/fa,2)*error2


###########################    DECAY TO  a-> pi pi gamma    ###########################
def ampatogammapipi(ma, Gamma, mrho, arhorho, x):
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta)
        #model: Coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a-> eta pi pi (without prefactor)
    if ma > 2*mpi+0.001:
        a = g**2* np.sqrt(x[0])* bw(np.sqrt(x[0]), mrho, Gamma,1)*arhorho*ffunction(ma)
        integrand = np.abs(a*np.conjugate(a)*pow(1-x[0]/ma**2,3)*pow(1-4*mpi**2/x[0],3/2))
    else:
        integrand = 0
    return integrand


def atogammapipi(M, m1, m2, m3, fa, Gamma, arhorho):
    #INPUT
        #M: Mass of decaying particle (in GeV) [ALP]
        #mi: Mass of daughter particle (in GeV) [pi, pi, photon]
        #fa: Scale of U(1)PQ (in GeV)
        #Gamma: Decay width of rho meson (in GeV)
        #arhorho: MMixing coupling arhorho
    #OUTPUT
        #decayrate: Decay rate
        #edecayrate: Error in decay rate
    
    if M > 2*mpi:
        #Numerical integration (using vegas integrator)
        integrator= vegas.Integrator([[(m1+m2)**2,(M-m3)**2]])#,[0,1]]) #Second integration is to get mean value easily
        # step 1 -- adapt to integrand; discard results
        integrator(functools.partial(ampatogammapipi, M, Gamma, mrho, arhorho), nitn=10, neval=1000)
        # step 2 -- integrator has adapted to integrand; keep results
        resint = integrator(functools.partial(ampatogammapipi, M, Gamma, mrho, arhorho), nitn=10, neval=1000)
        decayrate = 3*alphaem(M)*M**3/(2**11*np.pi**6*fa**2)* resint.mean 
        edecayrate = 3*alphaem(M)*M**3/(2**11*np.pi**6*fa**2)* resint.sdev
    else: decayrate, edecayrate= [0,0]
    return decayrate, edecayrate



######################################################   GLUON CHANNEL (a-> g g)    ######################################################
def atogluongluon(ma, fa):
    if ma > 1.84:
        res = alphas(ma)**2*ma**3/(32*np.pi**3*fa**2)* (1 + 83*alphas(ma)/(4*np.pi))
    else: res = 0 
    return res
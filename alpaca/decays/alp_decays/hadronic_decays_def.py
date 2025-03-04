import numpy as np
import flavio
import vegas as vegas
import functools
from . import threebody_decay
from ...rge import ALPcouplings, bases_above
from . import chiral
from .chiral import ffunction
from .u3reprs import pi0, eta, etap, rho0, omega, phi, sigma, f0, a0, f2, eta0, eta8
from ...constants import mu, md, ms, mc, mb, mt, me, mmu, mtau, mpi0, meta, metap, mK, mrho, fpi, mpi_pm, ma0, msigma, mf0, mf2, momega, Gammaa0, Gammasigma, Gammaf0, Gammaf2, Gammarho
from ...biblio.biblio import citations

#ALP decays to different channels (leptonic, hadronic, photons)


#Particle masses
mlepton = [me, mmu, mtau]
mqup = [mu, mc, mt]
mqdown = [md, ms, mb]
mquark = [mqup,mqdown]

pars = flavio.default_parameters.get_central_all()
alphaem = lambda q: flavio.physics.running.running.get_alpha_e(pars, q)
alphas = lambda q: flavio.physics.running.running.get_alpha_s(pars, q)
g=np.sqrt(12*np.pi)#0.6 


######################################################   HADRONIC CHANNELS    ######################################################
#Extracted from 1811.03474
#Following U(3) representation of ALPs 

###########################    ALP mixing   ###########################
def alp_mixing(M, fa):
    #INPUT:
        #M: Vector of matrices to multiply
    #OUTPUT:
        #trace of multiplied matrices
    qaux = np.identity(M[0].shape[0])
    for ii in range(len(M)):
        qaux = np.dot(qaux, M[ii])
    return 2*fa/fpi*np.trace(qaux)


def alpVV(ma: float, c: ALPcouplings, fa:float, **kwargs) -> tuple[float, float, float, float]:
    #INPUT:
        #ma: Mass of the ALP (GeV)
        #c: Vector of couplings (cu, cd, cs, cg)
    #OUTPUT:
        #<a rho rho>: Mixing element
    citations.register_inspire('Aloni:2018vki')
    aU3 = chiral.a_U3_repr(ma, c, fa, **kwargs)
    arhorho = alp_mixing([aU3, rho0, rho0], fa)
    arhow = alp_mixing([aU3, rho0, omega], fa)
    aww = alp_mixing([aU3, omega, omega], fa)
    aphiphi = alp_mixing([aU3, phi, phi], fa)
    return arhorho, arhow, aww, aphiphi

def G(x, y, z):
    return x*y + x*z + y*z

#ALP-meson mixing
    #Elements of the mass mixing matrix
Mpipi2 = mpi0**2 
Mpieta2 = - Mpipi2*np.sqrt(2/3)
Mpietap2 = - Mpipi2/np.sqrt(3)
Mapi2 = 0
Maeta2 = -np.sqrt(2/3)*mpi0/(mu+md)*mu*md*ms/G(mu,md,ms)
Maetap2 = Maeta2*2*np.sqrt(2)
Metaetap2 = 0


#Spp coefficients
def Spp(mP,mPp,MPPp2): #Eq. S8
    return MPPp2/(mP**2-mPp**2)

Setapi0 = Spp(meta,mpi0,Mpieta2)
Setappi0 = Spp(metap,mpi0,Mpietap2)
Setapeta = Spp(metap,meta,Metaetap2)



###########################    BREIT-WIGNER DISTRIBUTION   ###########################
#Breit-Wigner expression (from Eqs. 27-33 of PHYSICAL REVIEW D 86, 032013 (2012), arXiv:1205.2228)
def beta(x):
    #INPUT:
        #x: Energy^2 (in GeV)
    #OUTPUT:
        #beta function
    return np.sqrt(1-4*mpi0**2/x)

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
    return 3/np.pi* mpi0**2/kaux**2* np.log((m+2*kaux)/(2*mpi0))+ m/(2*np.pi*kaux)- mpi0**2*m/(np.pi*kaux**3)

def faux(x):
    #INPUT:
        #x: CM energy (in GeV^2)
    #OUTPUT:
        #kam: Auxiliary function 1
        #ham: Auxiliary function 2
        #hpam: Derivative of auxiliary function 2
    kam = 1/2*np.sqrt(x)*beta(x)
    ham = 2/np.pi * kam/np.sqrt(x)* np.log((np.sqrt(x)+2*kam)/(2*mpi0))
    hpam = (-4*mpi0**2+x+np.sqrt(x*(x-4*mpi0**2))+4*mpi0**2*(1+np.sqrt(1-4*mpi0**2/x))*np.log((np.sqrt(x)+np.sqrt(x-4*mpi0**2))/(2*mpi0)))/\
        (2*np.pi*x* (-4*mpi0**2+ x+ np.sqrt(x*(x-4*mpi0**2)))) #Obtained deriving with Mathematica
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
    citations.register_inspire('BaBar:2012bdw')
    if c==0:
        result= m**2/(m**2 -s -1.j*m*Gamma)
    elif c==1:
        result = m**2*(1+d(m)*Gamma/m)/(m**2-s+f(s, m, Gamma)-1.j*m*gammaf(s,m,Gamma))
    else: 
        print('Wrong control digit in BW function')
        result = 0
    return result

###########################    DECAY TO 3 PIONS a-> pi pi pi    ###########################
#Decay to 3 neutral pions 3 pi0
def ampato3pi0(ma, model, fa, Ener3, **kwargs): #Eq. S31
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV)
        #model: Model coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a->3 pi0 (without prefactor)
    citations.register_inspire('Aloni:2018vki')
    deltaI = (md-mu)/(md+mu)
    aU3 = chiral.a_U3_repr(ma, model, fa, **kwargs)
    coef = [alp_mixing([aU3, pi0], fa),\
            alp_mixing([aU3, eta], fa),\
            alp_mixing([aU3, etap], fa)]
    #coef = u3rep(ma, model, deltaI)
    aux = coef[0] - deltaI*(1/np.sqrt(3)+np.sqrt(2)*Setapi0+Setappi0)*(np.sqrt(2)*coef[1]+coef[2]) + \
    np.sqrt(3)*model['cg']*deltaI*(mpi0**2-2*meta**2)/(mpi0**2-4*meta**2)*(np.sqrt(2)*Setapi0+Setappi0)
    return mpi0**2*aux

#Decay to pi+ pi- pi0
def ampatopicharged(ma, model, fa, Ener3, **kwargs): #Eq.S32
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV)
        #model: Model coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a->3 pi0 (without prefactor)
    citations.register_inspire('Aloni:2018vki')
    deltaI = (md-mu)/(md+mu)
    aU3 = chiral.a_U3_repr(ma, model, fa, **kwargs)
    coef = [alp_mixing([aU3, pi0], fa),\
            alp_mixing([aU3, eta], fa),\
            alp_mixing([aU3, etap], fa)]
    mpipipm = Ener3
    aux = (3*mpipipm**2-ma**2-2*mpi_pm**2)*coef[0] - deltaI*mpi_pm**2*(1/np.sqrt(3)+np.sqrt(2)*Setapi0+Setappi0)*(np.sqrt(2)*coef[1]+coef[2]) + \
    model['cg']*deltaI*(mpi_pm**2-2*meta**2)/(mpi_pm**2-4*meta**2)*(np.sqrt(3)*mpi_pm**2*(np.sqrt(2)*Setapi0+Setappi0)-3*mpipipm**2+ma**2+3*mpi_pm**2)
    return 1/3*aux

#Decay rate (numerical integration)
k = 2.7 # Mean value of k-factor to reproduce masses of eta, eta'
def ato3pi(ma, m1, m2, m3, model, fa, c, **kwargs): #Eq. S33
    #INPUT:
        #ma: Mass of the ALP (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) --> Pions
        #fa: Scale of U(1)PQ (in GeV)
        #c: Control value (c=0-> Neutral pions, c=1-> pi0, pi+, pi-)
    #OUTPUT: 
        #Decay rate including symmetry factors
    citations.register_inspire('Aloni:2018vki')
    if c == 0:
        s = 3*2 #Symmetry factor
        if ma > 3*mpi0+0.001 and ma<metap: 
            result, error = threebody_decay.decay3body_spheric(ampato3pi0, ma, m1, m2, m3, model, fa, **kwargs) #Amplitude of decay to 3 neutral pions
        else: result, error = [0.0,0.0]
        #result2, error2 = threebody_decay2.decay3body(ampato3pi0, ma, m1, m2, m3) #Amplitude of decay to 3 neutral pions
    elif c == 1:
        s = 1 #Symmetry factor
        if ma > mpi0+2*mpi_pm+0.001 and ma<metap:  #mpi0+mpim+mpip
            result, error = threebody_decay.decay3body_spheric(ampatopicharged, ma, m1, m2, m3, model, fa, **kwargs) #Amplitude of decay to pi+ pi- pi0
        else: result, error = [0.0,0.0] 
        #result2, error2 = threebody_decay2.decay3body(ampato3pi0, ma, m1, m2, m3) #Amplitude of decay to 3 neutral pions
    return k/(2*ma*s)*1/pow(fpi*fa,2)*result, k/(2*ma*s)*1/pow(fpi*fa,2)*error#,k/(2*ma*s)*1/pow(fpi*fa,2)*result2, k/(2*ma*s)*1/pow(fpi*fa,2)*error2

def decay_width_3pi0pm(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return ato3pi(ma, mpi0, mpi_pm, mpi_pm, couplings, fa, 1, **kwargs)[0]

def decay_width_3pi000(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return ato3pi(ma, mpi0, mpi0, mpi0, couplings, fa, 0, **kwargs)[0]

###########################    DECAY TO  a-> eta pi pi    ###########################
#It is assumed that Fpppp(m)=Fspp(m)=Ftpp(m)=F(m)

def ampatoetapipi(ma, m1, m2, m3, model, fa, x, kinematics, **kwargs):
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

    citations.register_inspire('Aloni:2018vki')
    deltaI = 0
    #Kinematic relations
    mpi1pi2 = np.sqrt(ma**2+m3**2-2*ma*x[:,0])
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

    aU3 = chiral.a_U3_repr(ma, model, fa, **kwargs)

    ####DECAY a->eta pi pi
    aetasigma = alp_mixing([aU3, eta, sigma], fa)
    aetaf0 = alp_mixing([aU3, eta, f0], fa)
    api0a0 = alp_mixing([aU3, eta, a0], fa)
    aetaf2 = alp_mixing([aU3, eta, f2], fa)

    ####DECAY a->etaprime pi pi
    aetapsigma = alp_mixing([aU3, etap, sigma], fa)
    aetapf0 = alp_mixing([aU3, etap, f0], fa)
    aetapf2 = alp_mixing([aU3, etap, f2], fa)


    #Mix amplitude (Eq.S48)
    #amix = 0 #Approximation --> (np.sqrt(2)*aeta0+aeta8)*mpi**2/(3*fpi**2)*ffunction(ma) 
    amix = (np.sqrt(2)*alp_mixing([aU3, eta0], fa) + alp_mixing([aU3, eta8], fa))*m2**2/3/fpi**2*ffunction(ma)

    #a-> Sigma (Eq.S49)
    asigma = np.where(mpi1pi2 < 2*mK, -(10)**2* aetasigma* papeta* ppi1ppi2*bw(mpi1pi2**2, msigma, Gammasigma, 0)*ffunction(ma), 0)

    #a-> f0 (Eq.S50)
    af0 = (7.3)**2* aetaf0* papeta* ppi1ppi2* bw(mpi1pi2**2, mf0, Gammaf0, 0)*ffunction(ma) 

    #a-> a0 (Eq.S51)
    aa0 = (13)**2* api0a0* ffunction(ma)* (pappi2* petappi1*bw(metapi1**2, ma0, Gammaa0, 0) + pappi1* petappi2* bw(metapi2, ma0, Gammaa0,0)) 

    #a-> f2 (Eq.)
    af2 = (16)**2* aetaf2* (petaqpipi**2 - 1/3*qpipi2*(m3**2+ petappipi**2/ppi1ppi2**2- 2*petappipi**2/ppi1ppi2**2))*\
          bw(mpi1pi2**2, mf2, Gammaf2, 0)* ffunction(ma)
    aux = (amix+asigma+af0+aa0+af2) 
    return aux

def atoetapipi(ma, m1, m2, m3, model, fa, c, **kwargs): #Eq. S33
    #INPUT:
        #ma: Mass of the ALP (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta)
        #fa: Scale of U(1)PQ (in GeV)
        #c: Control value (c=0-> Neutral pions, c=1-> pi0, pi+, pi-)
    #OUTPUT: 
        #Decay rate including symmetry factors
    citations.register_inspire('Aloni:2018vki')
    if ma < m1 + m2 + m3:
        return [0.0, 0,0]
    s = 2-c # Symmetry factor: 2 for pi0 pi0, 1 for pi+ pi-
    result, error = threebody_decay.decay3body(ampatoetapipi, ma, m1, m2, m3, model, fa, **kwargs)
    return 1/(2*ma*s)*pow(fpi/fa,2)*result, 1/(2*ma*s)*pow(fpi/fa,2)*error

def decay_width_etapipi00(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetapipi(ma, meta, mpi0, mpi0, couplings, fa, 0, **kwargs)[0]

def decay_width_etapipipm(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetapipi(ma, meta, mpi_pm, mpi_pm, couplings, fa, 1, **kwargs)[0]

###########################    DECAY TO  a-> etap pi pi    ###########################
#It is assumed that Fpppp(m)=Fspp(m)=Ftpp(m)=F(m)

def ampatoetappipi(ma, m1, m2, m3, model, fa, x, kinematics, **kwargs):
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta')
        #model: Coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a-> eta pi pi (without prefactor)

    citations.register_inspire('Aloni:2018vki')
    deltaI = 0
    #Kinematic relations
    mpi1pi2 = np.sqrt(ma**2+m3**2-2*ma*x[:,0])
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

    aU3 = chiral.a_U3_repr(ma, model, fa, **kwargs)

    ####DECAY a->eta pi pi
    aetapsigma = alp_mixing([aU3, etap, sigma], fa)
    aetapf0 = alp_mixing([aU3, etap, f0], fa)
    api0a0p = alp_mixing([aU3, etap, a0], fa)
    aetapf2 = alp_mixing([aU3, etap, f2], fa)

    ####DECAY a->etaprime pi pi
    aetapsigma = alp_mixing([aU3, etap, sigma], fa)
    aetapf0 = alp_mixing([aU3, etap, f0], fa)
    aetapf2 = alp_mixing([aU3, etap, f2], fa)


    #Mix amplitude (Eq.S48)
    #amix = 0 #Approximation --> (np.sqrt(2)*aeta0+aeta8)*mpi**2/(3*fpi**2)*ffunction(ma) 
    amix = (np.sqrt(2)*alp_mixing([aU3, eta0], fa) + alp_mixing([aU3, eta8], fa))*m2**2/3/fpi**2*ffunction(ma)

    #a-> Sigma (Eq.S49)
    asigma = np.where(mpi1pi2 < 2*mK, -(10)**2* aetapsigma* papeta* ppi1ppi2*bw(mpi1pi2**2, msigma, Gammasigma, 0)*ffunction(ma), 0)

    #a-> f0 (Eq.S50)
    af0 = (7.3)**2* aetapf0* papeta* ppi1ppi2* bw(mpi1pi2**2, mf0, Gammaf0, 0)*ffunction(ma) 

    #a-> a0 (Eq.S51)
    aa0 = (13)**2* api0a0p* 1.2* ffunction(ma)* (pappi2* petappi1*bw(metapi1**2, ma0, Gammaa0, 0) + pappi1* petappi2* bw(metapi2, ma0, Gammaa0,0)) 

    #a-> f2 (Eq.)
    af2 = (16)**2* aetapf2* (petaqpipi**2 - 1/3*qpipi2*(m3**2+ petappipi**2/ppi1ppi2**2- 2*petappipi**2/ppi1ppi2**2))*\
          bw(mpi1pi2**2, mf2, Gammaf2, 0)* ffunction(ma)
    aux = (amix+asigma+af0+aa0+af2) 
    return aux

def atoetappipi(ma, m1, m2, m3, model, fa, c, **kwargs): #Eq. S33
    #INPUT:
        #ma: Mass of the ALP (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta)
        #fa: Scale of U(1)PQ (in GeV)
        #c: Control value (c=0-> Neutral pions, c=1-> pi0, pi+, pi-)
    #OUTPUT: 
        #Decay rate including symmetry factors
    citations.register_inspire('Aloni:2018vki')
    if ma < m1 + m2 + m3:
        return [0.0, 0,0]
    s = 2-c # Symmetry factor: 2 for pi0 pi0, 1 for pi+ pi-
    result, error = threebody_decay.decay3body(ampatoetappipi, ma, m1, m2, m3, model, fa, **kwargs)
    return 1/(2*ma*s)*pow(fpi/fa,2)*result, 1/(2*ma*s)*pow(fpi/fa,2)*error

def decay_width_etappipi00(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetappipi(ma, metap, mpi0, mpi0, couplings, fa, 0, **kwargs)[0]

def decay_width_etappipipm(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetappipi(ma, metap, mpi_pm, mpi_pm, couplings, fa, 1, **kwargs)[0]


###########################    DECAY TO  a-> pi pi gamma    ###########################
def ampatogammapipi(ma, Gamma, mrho, model, fa, x, **kwargs):
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta)
        #model: Coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a-> eta pi pi (without prefactor)
    citations.register_inspire('Aloni:2018vki')
    arhorho = alpVV(ma, model, fa, **kwargs)[0]
    if ma > 2*mpi0+0.001:
        a = g**2* np.sqrt(x[:,0])* bw(np.sqrt(x[:,0]), mrho, Gamma,1)*arhorho*ffunction(ma)
        integrand = np.abs(a*np.conjugate(a)*pow(1-x[:,0]/ma**2,3)*pow(1-4*mpi0**2/x[:,0],3/2))
    else:
        integrand = 0.0
    return integrand


def decay_width_gammapipi(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    #INPUT
        #M: Mass of decaying particle (in GeV) [ALP]
        #mi: Mass of daughter particle (in GeV) [pi, pi, photon]
        #model: Coupling of model studied
        #fa: Scale of U(1)PQ (in GeV)
        #Gamma: Decay width of rho meson (in GeV)
        #arhorho: Mixing coupling arhorho
    #OUTPUT
        #decayrate: Decay rate
        #edecayrate: Error in decay rate
    
    citations.register_bibtex('vegas', """@software{peter_lepage_2024_12687656,
  author       = {Peter Lepage},
  title        = {gplepage/vegas: vegas version 6.1.3},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v6.1.3},
  doi          = {10.5281/zenodo.12687656},
  url          = {https://doi.org/10.5281/zenodo.12687656}
}""")
    citations.register_inspire('Lepage:2020tgj')
    if ma > 2*mpi_pm:
        nitn_adapt = kwargs.get('nitn_adapt', 10)
        neval_adapt = kwargs.get('neval_adapt', 10)
        nitn = kwargs.get('nitn', 10)
        neval = kwargs.get('neval', 100)
        cores = kwargs.get('cores', 1)
        kwargs_integrand = {k: v for k, v in kwargs.items() if k not in ['nitn_adapt', 'neval_adapt', 'nitn', 'neval', 'cores']}
        #Numerical integration (using vegas integrator)
        integrator= vegas.Integrator([[(2*mpi_pm)**2,ma**2]], nproc=cores)#,[0,1]]) #Second integration is to get mean value easily
        # step 1 -- adapt to integrand; discard results
        integrator(vegas.lbatchintegrand(functools.partial(ampatogammapipi, ma, Gammarho, mrho, couplings, fa, **kwargs_integrand)), nitn=nitn_adapt, neval=neval_adapt)
        # step 2 -- integrator has adapted to integrand; keep results
        resint = integrator(vegas.lbatchintegrand(functools.partial(ampatogammapipi, ma, Gammarho, mrho, couplings, fa, **kwargs_integrand)), nitn=nitn, neval=neval)
        decayrate = 3*alphaem(ma)*ma**3/(2**11*np.pi**6*fa**2)* resint.mean 
        edecayrate = 3*alphaem(ma)*ma**3/(2**11*np.pi**6*fa**2)* resint.sdev
    else: decayrate, edecayrate= [0.0,0.0]
    return decayrate


def decay_width_2w(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    citations.register_inspire('Aloni:2018vki')
    if ma > 2*momega:
        aU3 = chiral.a_U3_repr(ma, couplings, fa, **kwargs)
        aww = alp_mixing([aU3, omega, omega], fa)
        aux = g**2*ffunction(ma)*aww
        decayrate = 9*ma**3/((4*np.pi)**5*fa**2)*(1-4*momega**2/ma**2)**(3/2)*np.abs(aux)**2
    else: decayrate= 0.0
    return decayrate
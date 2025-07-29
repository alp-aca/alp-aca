import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import flavio
import vegas as vegas
import functools
from . import threebody_decay
from ...rge import ALPcouplings, bases_above
from ...chiPT.chiral import ffunction, kappa, mesonmass_chiPT, sm_mixingangles, cqhat, a_U3_repr, mass_mixing
from ...chiPT.u3reprs import pi0, eta, etap, rho0, omega, phi, sigma, f0, a0, f2, eta0, eta8
from ...constants import mu, md, ms, mc, mb, mt, me, mmu, mtau, mpi0, meta, metap, mK, mrho, fpi, mpi_pm, ma0, msigma, mf0, mf2, momega, Gammaa0, Gammasigma, Gammaf0, Gammaf2, Gammarho, theta_eta_etap, mrho_pm, Gammarho_pm, ma0_pm, Gammaa0_pm
from ...biblio.biblio import citations
from ...chiPT.svt_lagrangians import gTf2, CoefA, CoefB, CoefC, CoefD, theta_S

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
        #c: Vector of couplings (cu, cd, cs, cG)
    #OUTPUT:
        #<a rho rho>: Mixing element
    citations.register_inspire('Aloni:2018vki')
    aU3 = a_U3_repr(ma, c, fa, **kwargs)
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

UnitStep = lambda x: np.heaviside(x, 0.0) #Heaviside function

###########################    DECAY TO 3 PIONS a-> pi pi pi    ###########################
#Decay to 3 neutral pions 3 pi0
def ampato3pi0(ma, m1, m2, m3, model, fa, x, kinematics, **kwargs): #Eq. S31
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV)
        #model: Model coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a->3 pi0 (without prefactor)
    citations.register_inspire('Aloni:2018vki')
    m12 = ma**2 + m3**2 -2*ma*x[:,0]
    m23 = ma**2 + m1**2 -2*kinematics[0]
    kappau = kappa[0,0]
    kappad = kappa[1,1]
    deltaI = (md-mu)/(md+mu)
    F0 = fpi/np.sqrt(2)
    cG = model['cG']*F0/fa
    aU3 = a_U3_repr(ma, model, fa, **kwargs)
    thpiALP = np.trace(np.dot(aU3, pi0))*2
    thetaa = np.trace(np.dot(aU3, eta))*2
    thetap = np.trace(np.dot(aU3, etap))*2
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    sm_angles = sm_mixingangles()
    thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
    thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap
    thetapi = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)* sm_angles[('eta', 'pi0')]
    thetaprpi = (s_eta+np.sqrt(2)*c_eta)*sm_angles[('etap', 'pi0')]
    kfact = 2.7 # Mean value of k-factor to reproduce decay rates of eta, eta'
    I = 1j

    amptot = -np.sqrt(kfact)*mpi0**2*(3*cG*(kappad*(np.sqrt(6)*deltaI*thetapi + np.sqrt(3)*deltaI*thetaprpi + deltaI + 1) + kappau*(np.sqrt(6)*deltaI*thetapi + np.sqrt(3)*deltaI*thetaprpi + deltaI - 1)) + deltaI*thetaALP*(6*thetapi + 3*np.sqrt(2)*thetaprpi + np.sqrt(6)) + deltaI*thetaprALP*(3*np.sqrt(2)*thetapi + 3*thetaprpi + np.sqrt(3)) - 3*thpiALP)*UnitStep(-ma + metap)/(3*F0**2) + 0j

    amptot += gTf2**2*((m12**2*(mf2**2 - 2*mpi0**2)*(ma**2 - mf2**2 + mpi0**2) + m12*(2*m23*(ma**2*(mf2**2 - 2*mpi0**2) + 2*mf2**4 + 3*mf2**2*mpi0**2 - 2*mpi0**4) - ma**4*(mf2**2 - 2*mpi0**2) + ma**2*(mf2**4 - 16*mf2**2*mpi0**2 + 20*mpi0**4) + 3*mf2**4*mpi0**2 - 19*mf2**2*mpi0**4 + 10*mpi0**6) + m23**2*(mf2**2 - 2*mpi0**2)*(ma**2 - mf2**2 + mpi0**2) + m23*(-ma**4*(mf2**2 - 2*mpi0**2) + ma**2*(mf2**4 - 16*mf2**2*mpi0**2 + 20*mpi0**4) + 3*mf2**4*mpi0**2 - 19*mf2**2*mpi0**4 + 10*mpi0**6) - 2*mpi0**2*(-3*ma**4*(mf2**2 - 2*mpi0**2) + 4*ma**2*(mf2**4 - 6*mf2**2*mpi0**2 + 5*mpi0**4) + 4*mf2**4*mpi0**2 - 13*mf2**2*mpi0**4 + 6*mpi0**6))*UnitStep(-m12 - m23 + ma**2 + 3*mpi0**2 - (Gammaf2 - mf2)**2)/(-4*I*Gammaf2*mf2 + 4*m12 + 4*m23 - 4*ma**2 + 4*mf2**2 - 12*mpi0**2) + (-6*m12**2*mf2**4 + 6*m12*mf2**4*(ma**2 + 3*mpi0**2) + m23**2*(mf2**2 - 2*mpi0**2)*(ma**2 - mf2**2 + mpi0**2) + m23*(-6*m12*mf2**4 - ma**4*(mf2**2 - 2*mpi0**2) + ma**2*(mf2**4 + 4*mf2**2*mpi0**2 - 4*mpi0**4) + 3*mf2**4*mpi0**2 + mf2**2*mpi0**4 + 2*mpi0**6) - 4*mf2**2*mpi0**2*(ma**4 + 2*ma**2*(mf2**2 - mpi0**2) + 2*mf2**2*mpi0**2 + mpi0**4))*UnitStep(m23 - (Gammaf2 - mf2)**2)/(-4*m23 + 4*mf2*(-I*Gammaf2 + mf2)) + (m12**2*(mf2**2 - 2*mpi0**2)*(ma**2 - mf2**2 + mpi0**2) + m12*(-6*m23*mf2**4 - ma**4*(mf2**2 - 2*mpi0**2) + ma**2*(mf2**4 + 4*mf2**2*mpi0**2 - 4*mpi0**4) + 3*mf2**4*mpi0**2 + mf2**2*mpi0**4 + 2*mpi0**6) - 6*m23**2*mf2**4 + 6*m23*mf2**4*(ma**2 + 3*mpi0**2) - 4*mf2**2*mpi0**2*(ma**4 + 2*ma**2*(mf2**2 - mpi0**2) + 2*mf2**2*mpi0**2 + mpi0**4))*UnitStep(m12 - (Gammaf2 - mf2)**2)/(-4*m12 + 4*mf2*(-I*Gammaf2 + mf2)))*(3*cG*(kappad - kappau) + deltaI*thetaALP*(2*thetapi + np.sqrt(2)*thetaprpi) + deltaI*(np.sqrt(2)*thetapi + thetaprpi)*(np.sqrt(3)*cG*(kappad + kappau) + thetaprALP) - 3*thpiALP)/(36*mf2**4)

    amptot += -np.sqrt(3)*deltaI*(cG*(2*CoefA*(kappad*(np.sqrt(6)*deltaI*thetapi - 2*np.sqrt(3)*deltaI*thetaprpi + 6) + kappau*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + 6) - 6) + CoefC*(np.sqrt(3)*deltaI*(kappad - kappau)*(np.sqrt(2)*thetapi + 4*thetaprpi) + 6)) + np.sqrt(3)*(2*CoefA*(deltaI*thpiALP*(-np.sqrt(2)*thetapi + 2*thetaprpi) + np.sqrt(2)*thetaALP - 2*thetaprALP) + CoefC*(-deltaI*thpiALP*(np.sqrt(2)*thetapi + 4*thetaprpi) + np.sqrt(2)*thetaALP + 4*thetaprALP)))*(2*np.sqrt(2)*CoefA*thetapi - 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi + 4*CoefC*thetaprpi)*(-2*Gammaa0**2*ma0**2*(m23**2 - m23*(ma**2 + 3*mpi0**2) + 3*mpi0**2*(ma**2 + mpi0**2)) + I*Gammaa0*ma0*(m23**2*(ma**2 - 4*ma0**2 + 3*mpi0**2) - m23*(ma**2 + 3*mpi0**2)*(ma**2 - 4*ma0**2 + 3*mpi0**2) + 4*mpi0**2*(ma**2 + mpi0**2)*(ma**2 - 3*ma0**2 + 3*mpi0**2)) + m12**2*(-2*Gammaa0**2*ma0**2 + I*Gammaa0*ma0*(ma**2 - 4*ma0**2 + 3*mpi0**2) + m23*(3*I*Gammaa0*ma0 + 2*ma**2 - 3*ma0**2 + 6*mpi0**2) - ma**2*ma0**2 - 2*ma**2*mpi0**2 + 2*ma0**4 - 3*ma0**2*mpi0**2 - 2*mpi0**4) + m12*(m23**2*(3*I*Gammaa0*ma0 + 2*ma**2 - 3*ma0**2 + 6*mpi0**2) - 2*m23*(Gammaa0**2*ma0**2 + I*Gammaa0*(2*ma0**3 + 3*ma0*mpi0**2) + ma**4 + ma**2*(I*Gammaa0*ma0 - ma0**2 + 7*mpi0**2) - ma0**4 - 3*ma0**2*mpi0**2 + 10*mpi0**4) + (ma**2 + 3*mpi0**2)*(2*Gammaa0**2*ma0**2 - I*Gammaa0*ma0*(ma**2 - 4*ma0**2 + 3*mpi0**2) + ma**2*(ma0**2 + 2*mpi0**2) - 2*ma0**4 + 3*ma0**2*mpi0**2 + 2*mpi0**4)) - m23**2*ma**2*ma0**2 - 2*m23**2*ma**2*mpi0**2 + 2*m23**2*ma0**4 - 3*m23**2*ma0**2*mpi0**2 - 2*m23**2*mpi0**4 + m23*ma**4*ma0**2 + 2*m23*ma**4*mpi0**2 - 2*m23*ma**2*ma0**4 + 6*m23*ma**2*ma0**2*mpi0**2 + 8*m23*ma**2*mpi0**4 - 6*m23*ma0**4*mpi0**2 + 9*m23*ma0**2*mpi0**4 + 6*m23*mpi0**6 - 4*ma**4*ma0**2*mpi0**2 + 6*ma**2*ma0**4*mpi0**2 - 16*ma**2*ma0**2*mpi0**4 + 6*ma0**4*mpi0**4 - 12*ma0**2*mpi0**6)/(72*(m12 + I*ma0*(Gammaa0 + I*ma0))*(m23 + I*ma0*(Gammaa0 + I*ma0))*(-I*Gammaa0*ma0 + m12 + m23 - ma**2 + ma0**2 - 3*mpi0**2))

    amptot += I*(2*CoefB*np.cos(theta_S) + np.sqrt(2)*(-CoefA + CoefB)*np.sin(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.cos(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.sin(theta_S))*(-2*Gammaf0**2*mf0**2*(m23**2 - m23*(ma**2 + 3*mpi0**2) + 3*mpi0**2*(ma**2 + mpi0**2)) + I*Gammaf0*mf0*(m23**2*(ma**2 - 4*mf0**2 + 3*mpi0**2) - m23*(ma**2 + 3*mpi0**2)*(ma**2 - 4*mf0**2 + 3*mpi0**2) + 4*mpi0**2*(ma**2 + mpi0**2)*(ma**2 - 3*mf0**2 + 3*mpi0**2)) + m12**2*(-2*Gammaf0**2*mf0**2 + I*Gammaf0*mf0*(ma**2 - 4*mf0**2 + 3*mpi0**2) + m23*(3*I*Gammaf0*mf0 + 2*ma**2 - 3*mf0**2 + 6*mpi0**2) - ma**2*mf0**2 - 2*ma**2*mpi0**2 + 2*mf0**4 - 3*mf0**2*mpi0**2 - 2*mpi0**4) + m12*(m23**2*(3*I*Gammaf0*mf0 + 2*ma**2 - 3*mf0**2 + 6*mpi0**2) - 2*m23*(Gammaf0**2*mf0**2 + I*Gammaf0*(2*mf0**3 + 3*mf0*mpi0**2) + ma**4 + ma**2*(I*Gammaf0*mf0 - mf0**2 + 7*mpi0**2) - mf0**4 - 3*mf0**2*mpi0**2 + 10*mpi0**4) + (ma**2 + 3*mpi0**2)*(2*Gammaf0**2*mf0**2 - I*Gammaf0*mf0*(ma**2 - 4*mf0**2 + 3*mpi0**2) + ma**2*(mf0**2 + 2*mpi0**2) - 2*mf0**4 + 3*mf0**2*mpi0**2 + 2*mpi0**4)) - m23**2*ma**2*mf0**2 - 2*m23**2*ma**2*mpi0**2 + 2*m23**2*mf0**4 - 3*m23**2*mf0**2*mpi0**2 - 2*m23**2*mpi0**4 + m23*ma**4*mf0**2 + 2*m23*ma**4*mpi0**2 - 2*m23*ma**2*mf0**4 + 6*m23*ma**2*mf0**2*mpi0**2 + 8*m23*ma**2*mpi0**4 - 6*m23*mf0**4*mpi0**2 + 9*m23*mf0**2*mpi0**4 + 6*m23*mpi0**6 - 4*ma**4*mf0**2*mpi0**2 + 6*ma**2*mf0**4*mpi0**2 - 16*ma**2*mf0**2*mpi0**4 + 6*mf0**4*mpi0**4 - 12*mf0**2*mpi0**6)/(24*(m12 + I*mf0*(Gammaf0 + I*mf0))*(-I*m23 + mf0*(Gammaf0 + I*mf0))*(-I*Gammaf0*mf0 + m12 + m23 - ma**2 + mf0**2 - 3*mpi0**2))

    amptot += (2*CoefB*np.sin(theta_S) + np.sqrt(2)*(CoefA - CoefB)*np.cos(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.sin(theta_S) - 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.cos(theta_S))*((m12 - 2*mpi0**2)*(m12 - ma**2 - mpi0**2)*UnitStep(-m12 + 4*mK**2)/(m12 + I*msigma*(Gammasigma + I*msigma)) + (m23 - 2*mpi0**2)*(m23 - ma**2 - mpi0**2)*UnitStep(-m23 + 4*mK**2)/(m23 + I*msigma*(Gammasigma + I*msigma)) - (m12 + m23 - 2*mpi0**2)*(m12 + m23 - ma**2 - mpi0**2)*UnitStep(m12 + m23 + 4*mK**2 - ma**2 - 3*mpi0**2)/(-I*Gammasigma*msigma + m12 + m23 - ma**2 - 3*mpi0**2 + msigma**2))/24

    return amptot

#Decay to pi+ pi- pi0
def ampatopicharged(ma, m1, m2, m3, model, fa, x, kinematics, **kwargs): #Eq.S32
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV)
        #model: Model coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a->3 pi0 (without prefactor)
    citations.register_inspire('Aloni:2018vki')
    m12 = ma**2 + m3**2 -2*ma*x[:,0]
    m23 = ma**2 + m1**2 -2*kinematics[0]
    kappau = kappa[0,0]
    kappad = kappa[1,1]
    deltaI = (md-mu)/(md+mu)
    F0 = fpi/np.sqrt(2)
    cG = model['cG']*F0/fa
    aU3 = a_U3_repr(ma, model, fa, **kwargs)
    thpiALP = np.trace(aU3@pi0)*2
    thetaa = np.trace(aU3@eta)*2
    thetap = np.trace(aU3@etap)*2
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    sm_angles = sm_mixingangles()
    thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
    thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap
    thetapi = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)* sm_angles[('eta', 'pi0')]
    thetaprpi = (s_eta+np.sqrt(2)*c_eta)*sm_angles[('etap', 'pi0')]
    kfact = 2.7 # Mean value of k-factor to reproduce decay rates of eta, eta'
    cq = cqhat(model, ma, **kwargs)*F0/fa
    cuhat = cq[0,0]
    cdhat = cq[1,1]
    I = 1j

    amp_tot = 0j -np.sqrt(kfact)*(cdhat*(9*m12 - 3*ma**2 - 9*mpi0**2) + 6*cG*mpi0**2*(kappad*(np.sqrt(6)*deltaI*thetapi + np.sqrt(3)*deltaI*thetaprpi + deltaI + 1) + kappau*(np.sqrt(6)*deltaI*thetapi + np.sqrt(3)*deltaI*thetaprpi + deltaI - 1)) + cuhat*(-9*m12 + 3*ma**2 + 9*mpi0**2) + 2*deltaI*mpi0**2*(thetaALP*(6*thetapi + 3*np.sqrt(2)*thetaprpi + np.sqrt(6)) + thetaprALP*(3*np.sqrt(2)*thetapi + 3*thetaprpi + np.sqrt(3))) + 6*thpiALP*(-3*m12 + ma**2 + 2*mpi0**2))*UnitStep(-ma + metap)/(18*F0**2)+0j

    amp_tot += gTf2**2*(3*cG*(kappad - kappau) + deltaI*thetaALP*(2*thetapi + np.sqrt(2)*thetaprpi) + deltaI*(np.sqrt(2)*thetapi + thetaprpi)*(np.sqrt(3)*cG*(kappad + kappau) + thetaprALP) - 3*thpiALP)*(m12**2*(mf2**2 - 2*mpi0**2)*(ma**2 - mf2**2 + mpi0**2) + m12*(-6*m23*mf2**4 - ma**4*(mf2**2 - 2*mpi0**2) + ma**2*(mf2**4 + 4*mf2**2*mpi0**2 - 4*mpi0**4) + 3*mf2**4*mpi0**2 + mf2**2*mpi0**4 + 2*mpi0**6) - 6*m23**2*mf2**4 + 6*m23*mf2**4*(ma**2 + 3*mpi0**2) - 4*mf2**2*mpi0**2*(ma**4 + 2*ma**2*(mf2**2 - mpi0**2) + 2*mf2**2*mpi0**2 + mpi0**4))*UnitStep(m12 - (Gammaf2 - mf2)**2)/(144*mf2**4*(-m12 + mf2*(-I*Gammaf2 + mf2)))

    amp_tot += mrho_pm**2*(cG*(kappad - kappau) - thpiALP)*(2*m12**2 + 2*m12*m23 - 3*m12*(ma**2 + 3*mpi0**2 + I*mrho_pm*(Gammarho_pm + I*mrho_pm)) + 2*m23**2 - 2*m23*(ma**2 + 3*mpi0**2) + (ma**2 + 3*mpi0**2)*(ma**2 + 3*mpi0**2 + I*mrho_pm*(Gammarho_pm + I*mrho_pm)))/(2*F0**2*(m23 + I*mrho_pm*(Gammarho_pm + I*mrho_pm))*(-I*Gammarho_pm*mrho_pm + m12 + m23 - ma**2 - 3*mpi0**2 + mrho_pm**2))

    amp_tot += -np.sqrt(3)*deltaI*(6*cG*(2*CoefA*(kappad + kappau - 1) + CoefC) + np.sqrt(3)*(2*np.sqrt(2)*CoefA*thetaALP - 4*CoefA*thetaprALP + np.sqrt(2)*CoefC*thetaALP + 4*CoefC*thetaprALP))*(m12**2*(m23 + I*ma0_pm*(Gammaa0_pm + I*ma0_pm)) + m12*(-I*Gammaa0_pm*ma0_pm*(ma**2 + 3*mpi0**2) + m23**2 + 2*I*m23*ma0_pm*(Gammaa0_pm + I*ma0_pm) + ma**2*ma0_pm**2 - 2*ma**2*mpi0**2 + 3*ma0_pm**2*mpi0**2 - 2*mpi0**4) + (m23 - 2*mpi0**2)*(m23 - ma**2 - mpi0**2)*(2*I*Gammaa0_pm*ma0_pm + ma**2 - 2*ma0_pm**2 + 3*mpi0**2))*(2*np.sqrt(2)*CoefA*thetapi - 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi + 4*CoefC*thetaprpi)/(144*(m23 + I*ma0_pm*(Gammaa0_pm + I*ma0_pm))*(-I*Gammaa0_pm*ma0_pm + m12 + m23 - ma**2 + ma0_pm**2 - 3*mpi0**2))

    amp_tot += (m12 - 2*mpi0**2)*(2*CoefB*np.cos(theta_S) + np.sqrt(2)*(-CoefA + CoefB)*np.sin(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.cos(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.sin(theta_S))*(m12 - ma**2 - mpi0**2)/(24*m12 + 24*I*mf0*(Gammaf0 + I*mf0))

    amp_tot += -(m12 - 2*mpi0**2)*(2*CoefB*np.sin(theta_S) + np.sqrt(2)*(CoefA - CoefB)*np.cos(theta_S))*(-(-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.sin(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.cos(theta_S))*(m12 - ma**2 - mpi0**2)*UnitStep(-m12 + 4*mK**2)/(24*(m12 + I*msigma*(Gammasigma + I*msigma)))

    return amp_tot


#Decay rate (numerical integration)
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
        if ma > 3*mpi0+0.001: 
            result, error = threebody_decay.decay3body(ampato3pi0, ma, m1, m2, m3, model, fa, **kwargs) #Amplitude of decay to 3 neutral pions
        else: result, error = [0.0,0.0]
        #result2, error2 = threebody_decay2.decay3body(ampato3pi0, ma, m1, m2, m3) #Amplitude of decay to 3 neutral pions
    elif c == 1:
        s = 1 #Symmetry factor
        if ma > mpi0+2*mpi_pm+0.001:  #mpi0+mpim+mpip
            result, error = threebody_decay.decay3body(ampatopicharged, ma, m1, m2, m3, model, fa, **kwargs) #Amplitude of decay to pi+ pi- pi0
        else: result, error = [0.0,0.0] 
        #result2, error2 = threebody_decay2.decay3body(ampato3pi0, ma, m1, m2, m3) #Amplitude of decay to 3 neutral pions
    return ffunction(ma)**2/(2*ma*s)*result, ffunction(ma)**2/(2*ma*s)*error#,k/(2*ma*s)*1/pow(fpi*fa,2)*result2, k/(2*ma*s)*1/pow(fpi*fa,2)*error2

def decay_width_3pi0pm(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return ato3pi(ma, mpi0, mpi_pm, mpi_pm, couplings, fa, 1, **kwargs)[0]

def decay_width_3pi000(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return ato3pi(ma, mpi0, mpi0, mpi0, couplings, fa, 0, **kwargs)[0]

###########################    DECAY TO  a-> eta pi pi    ###########################
#It is assumed that Fpppp(m)=Fspp(m)=Ftpp(m)=F(m)

def ampatoetapi0pi0(ma, m1, m2, m3, model, fa, x, kinematics, **kwargs):
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
    m12 = ma**2 + m3**2 -2*ma*x[:,0]
    m23 = ma**2 + m1**2 -2*kinematics[0]
    kappau = kappa[0,0]
    kappad = kappa[1,1]
    deltaI = (md-mu)/(md+mu)
    F0 = fpi/np.sqrt(2)
    cG = model['cG']*F0/fa
    aU3 = a_U3_repr(ma, model, fa, **kwargs)
    thpiALP = np.trace(np.dot(aU3, pi0))*2
    thetaa = np.trace(np.dot(aU3, eta))*2
    thetap = np.trace(np.dot(aU3, etap))*2
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    sm_angles = sm_mixingangles()
    thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
    thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap
    thetapi = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)* sm_angles[('eta', 'pi0')]
    thetaprpi = (s_eta+np.sqrt(2)*c_eta)*sm_angles[('etap', 'pi0')]
    I = 1j

    amp_tot = 0j + mpi0**2*(cG*kappad*(deltaI*(thetapi + 2*np.sqrt(2)*thetaprpi + np.sqrt(6)) + np.sqrt(6)) + cG*kappau*(-deltaI*(thetapi + 2*np.sqrt(2)*thetaprpi + np.sqrt(6)) + np.sqrt(6)) - deltaI*thetapi*thpiALP - 2*np.sqrt(2)*deltaI*thetaprpi*thpiALP - np.sqrt(6)*deltaI*thpiALP + 2*thetaALP + np.sqrt(2)*thetaprALP)/(3*F0**2)

    amp_tot += -deltaI*gTf2**2*(-thetapi + np.sqrt(2)*thetaprpi)*(-((m23 - 2*mpi0**2)*(-meta**2*(m12 + m23 - ma**2 + 3*mf2**2 - 5*mpi0**2) + (mf2**2 - mpi0**2)*(m12 + m23 - ma**2 + 3*mf2**2 - mpi0**2)) + (2*mpi0**2*(2*meta**2 - 3*mf2**2) + (meta**2 - mf2**2 + mpi0**2)*(-m12 - m23 + ma**2 + mpi0**2))*(m12 - meta**2 - mpi0**2))*(-m23 + ma**2 + meta**2) - ((m23 - 2*mpi0**2)*(-meta**2*(m12 + m23 - ma**2 + 6*mf2**2 - 5*mpi0**2) + (-mf2**2 + mpi0**2)*(-m12 - m23 + ma**2 + mpi0**2)) + (-meta**2*(m12 + m23 - ma**2 + 3*mf2**2 - 5*mpi0**2) + (mf2**2 - mpi0**2)*(m12 + m23 - ma**2 + 3*mf2**2 - mpi0**2))*(m12 - meta**2 - mpi0**2))*(-m12 + ma**2 + mpi0**2) + (4*meta**2*mpi0**2*(meta**2 - 2*mf2**2 + mpi0**2) + (meta**2 - mf2**2 + mpi0**2)*(m12 + m23 - ma**2 - mpi0**2)**2 + (-m12 - m23 + ma**2 + mpi0**2)*(meta**4 - 3*meta**2*(mf2**2 - 2*mpi0**2) + 2*mf2**4 - 3*mf2**2*mpi0**2 + mpi0**4))*(m12 + m23 - meta**2 - mpi0**2))*(3*cG*(kappad - kappau) + deltaI*thetaALP*(2*thetapi + np.sqrt(2)*thetaprpi) + deltaI*(np.sqrt(2)*thetapi + thetaprpi)*(np.sqrt(3)*cG*(kappad + kappau) + thetaprALP) - 3*thpiALP)*UnitStep(-m12 - m23 + ma**2 + meta**2 + 2*mpi0**2 - (Gammaf2 - mf2)**2)/(432*mf2**4*(I*Gammaf2*mf2 - m12 - m23 + ma**2 + meta**2 - mf2**2 + 2*mpi0**2)) - deltaI*gTf2**2*(thetapi - np.sqrt(2)*thetaprpi)*(cG*kappad*(np.sqrt(3)*deltaI*(np.sqrt(2)*thetapi + thetaprpi) + 3) + cG*kappau*(np.sqrt(3)*deltaI*(np.sqrt(2)*thetapi + thetaprpi) - 3) + deltaI*(2*thetaALP*thetapi + np.sqrt(2)*thetaALP*thetaprpi + np.sqrt(2)*thetapi*thetaprALP + thetaprALP*thetaprpi) - 3*thpiALP)*(m12**2*(ma**2 - mf2**2 + mpi0**2)*(meta**2 - mf2**2 + mpi0**2) - m12*(-6*m23*mf2**4 + ma**4*(meta**2 - mf2**2 + mpi0**2) + ma**2*(meta**4 - 2*meta**2*(mf2**2 + 2*mpi0**2) + mf2**4 + 6*mf2**2*mpi0**2 - mpi0**4) + meta**4*(-mf2**2 + mpi0**2) + meta**2*(mf2**4 + 6*mf2**2*mpi0**2 - mpi0**4) + 2*(-mf2**2*mpi0 + mpi0**3)**2) + 6*m23**2*mf2**4 - 6*m23*meta**2*mf2**4 - 6*m23*meta**2*mf2**2*mpi0**2 - 12*m23*mf2**4*mpi0**2 + 6*m23*mf2**2*mpi0**4 + ma**4*(meta**4 - meta**2*(mf2**2 + 2*mpi0**2) + 5*mf2**2*mpi0**2 + mpi0**4) + ma**2*(-6*m23*mf2**2*(-meta**2 + mf2**2 + mpi0**2) - meta**4*(mf2**2 + 2*mpi0**2) + meta**2*(mf2**4 - 8*mf2**2*mpi0**2 + 4*mpi0**4) + 7*mf2**4*mpi0**2 + mf2**2*mpi0**4 - 2*mpi0**6) + 5*meta**4*mf2**2*mpi0**2 + meta**4*mpi0**4 + 7*meta**2*mf2**4*mpi0**2 + meta**2*mf2**2*mpi0**4 - 2*meta**2*mpi0**6 + mf2**4*mpi0**4 - 2*mf2**2*mpi0**6 + mpi0**8)*UnitStep(m12 - (Gammaf2 - mf2)**2)/(432*mf2**4*(-m12 + mf2*(-I*Gammaf2 + mf2))) - gTf2**2*(m23**2*(mf2**2 - 2*mpi0**2)*(ma**2 + meta**2 - mf2**2) + m23*(-ma**4*(mf2**2 - 2*mpi0**2) + ma**2*(2*meta**2*(mf2**2 - 2*mpi0**2) + mf2**4 + 2*mf2**2*mpi0**2) - meta**4*(mf2**2 - 2*mpi0**2) + meta**2*(mf2**4 + 2*mf2**2*mpi0**2) + 2*mf2**4*(-3*m12 + mpi0**2)) - 2*mf2**2*(3*m12**2*mf2**2 - 3*m12*mf2**2*(ma**2 + meta**2 + 2*mpi0**2) + 2*ma**4*mpi0**2 + ma**2*(meta**2*(3*mf2**2 - 4*mpi0**2) + mf2**2*mpi0**2) + mpi0**2*(2*meta**4 + meta**2*mf2**2 + 3*mf2**2*mpi0**2)))*(cG*kappad*(-3*deltaI*thetapi + np.sqrt(6)) + cG*kappau*(3*deltaI*thetapi + np.sqrt(6)) + 3*deltaI*thetapi*thpiALP + 2*thetaALP + np.sqrt(2)*thetaprALP)*UnitStep(m23 - (Gammaf2 - mf2)**2)/(144*mf2**4*(-m23 + mf2*(-I*Gammaf2 + mf2)))

    amp_tot += np.sqrt(3)*(2*deltaI*(m23 - 2*mpi0**2)*(cG*(2*CoefA*(6*deltaI*kappad*thetapi + 6*deltaI*kappau*thetapi - 6*deltaI*thetapi - np.sqrt(6)*kappad + np.sqrt(6)*kappau) + CoefC*(6*deltaI*thetapi - np.sqrt(6)*kappad + np.sqrt(6)*kappau)) + np.sqrt(3)*(2*CoefA*deltaI*thetapi*(np.sqrt(2)*thetaALP - 2*thetaprALP) + 2*np.sqrt(2)*CoefA*thpiALP + CoefC*deltaI*thetapi*(np.sqrt(2)*thetaALP + 4*thetaprALP) + np.sqrt(2)*CoefC*thpiALP))*(m23 - ma**2 - meta**2)*(2*np.sqrt(2)*CoefA*thetapi - 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi + 4*CoefC*thetaprpi)/(m23 + I*ma0*(Gammaa0 + I*ma0)) + np.sqrt(2)*(2*CoefA + CoefC)*(cG*(2*CoefA*(kappad*(np.sqrt(6)*deltaI*thetapi - 2*np.sqrt(3)*deltaI*thetaprpi + 6) + kappau*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + 6) - 6) + CoefC*(np.sqrt(3)*deltaI*(kappad - kappau)*(np.sqrt(2)*thetapi + 4*thetaprpi) + 6)) + np.sqrt(3)*(2*CoefA*(deltaI*thpiALP*(-np.sqrt(2)*thetapi + 2*thetaprpi) + np.sqrt(2)*thetaALP - 2*thetaprALP) + CoefC*(-deltaI*thpiALP*(np.sqrt(2)*thetapi + 4*thetaprpi) + np.sqrt(2)*thetaALP + 4*thetaprALP)))*(m12 + m23 - ma**2 - mpi0**2)*(m12 + m23 - meta**2 - mpi0**2)/(-I*Gammaa0*ma0 + m12 + m23 - ma**2 + ma0**2 - meta**2 - 2*mpi0**2) - np.sqrt(2)*(2*CoefA + CoefC)*(cG*(2*CoefA*(kappad*(np.sqrt(6)*deltaI*thetapi - 2*np.sqrt(3)*deltaI*thetaprpi + 6) + kappau*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + 6) - 6) + CoefC*(np.sqrt(3)*deltaI*(kappad - kappau)*(np.sqrt(2)*thetapi + 4*thetaprpi) + 6)) + np.sqrt(3)*(2*CoefA*(deltaI*thpiALP*(-np.sqrt(2)*thetapi + 2*thetaprpi) + np.sqrt(2)*thetaALP - 2*thetaprALP) + CoefC*(-deltaI*thpiALP*(np.sqrt(2)*thetapi + 4*thetaprpi) + np.sqrt(2)*thetaALP + 4*thetaprALP)))*(m12 - ma**2 - mpi0**2)*(m12 - meta**2 - mpi0**2)/(m12 + I*ma0*(Gammaa0 + I*ma0)))/144

    amp_tot += deltaI*((-8*CoefA*thetapi + 2*np.sqrt(2)*CoefA*thetaprpi + 4*CoefC*thetapi + 5*np.sqrt(2)*CoefC*thetaprpi + 4*CoefD*thetapi + 8*np.sqrt(2)*CoefD*thetaprpi)*np.cos(theta_S) + 2*(5*np.sqrt(2)*CoefA*thetapi + 2*CoefA*thetaprpi - np.sqrt(2)*CoefC*thetapi - CoefC*thetaprpi + np.sqrt(2)*CoefD*thetapi + 4*CoefD*thetaprpi)*np.sin(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.cos(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.sin(theta_S))*(m12 + m23 - ma**2 - mpi0**2)*(m12 + m23 - meta**2 - mpi0**2)/(-144*I*Gammaf0*mf0 + 144*m12 + 144*m23 - 144*ma**2 - 144*meta**2 + 144*mf0**2 - 288*mpi0**2) - deltaI*((-8*CoefA*thetapi + 2*np.sqrt(2)*CoefA*thetaprpi + 4*CoefC*thetapi + 5*np.sqrt(2)*CoefC*thetaprpi + 4*CoefD*thetapi + 8*np.sqrt(2)*CoefD*thetaprpi)*np.cos(theta_S) + 2*(5*np.sqrt(2)*CoefA*thetapi + 2*CoefA*thetaprpi - np.sqrt(2)*CoefC*thetapi - CoefC*thetaprpi + np.sqrt(2)*CoefD*thetapi + 4*CoefD*thetaprpi)*np.sin(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.cos(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.sin(theta_S))*(m12 - ma**2 - mpi0**2)*(m12 - meta**2 - mpi0**2)/(144*(m12 + I*mf0*(Gammaf0 + I*mf0))) - (m23 - 2*mpi0**2)*(2*CoefB*np.cos(theta_S) + np.sqrt(2)*(-CoefA + CoefB)*np.sin(theta_S))*(2*(CoefA*(-3*np.sqrt(2)*deltaI*thetapi*thpiALP + 2*np.sqrt(2)*thetaALP + 2*thetaprALP) + 3*np.sqrt(2)*CoefB*deltaI*thetapi*thpiALP + 3*np.sqrt(2)*CoefB*thetaALP - np.sqrt(2)*CoefC*thetaALP - CoefC*thetaprALP + np.sqrt(2)*CoefD*thetaALP + 4*CoefD*thetaprALP + cG*(3*np.sqrt(2)*CoefA*deltaI*kappad*thetapi - 3*np.sqrt(2)*CoefA*deltaI*kappau*thetapi + 2*np.sqrt(3)*CoefA*kappad + 2*np.sqrt(3)*CoefA*kappau + CoefB*(-3*np.sqrt(2)*deltaI*kappad*thetapi + 3*np.sqrt(2)*deltaI*kappau*thetapi + 4*np.sqrt(3)*kappad + 4*np.sqrt(3)*kappau - 2*np.sqrt(3)) - np.sqrt(3)*CoefC*kappad - np.sqrt(3)*CoefC*kappau + 2*np.sqrt(3)*CoefD))*np.sin(theta_S) + (-8*CoefA*thetaALP + 2*np.sqrt(2)*CoefA*thetaprALP + 12*CoefB*deltaI*thetapi*thpiALP + 12*CoefB*thetaALP + 4*CoefC*thetaALP + 5*np.sqrt(2)*CoefC*thetaprALP + 4*CoefD*thetaALP + 8*np.sqrt(2)*CoefD*thetaprALP + cG*(-2*np.sqrt(6)*CoefA*(3*kappad + 3*kappau - 2) + 4*CoefB*(-3*deltaI*kappad*thetapi + 3*deltaI*kappau*thetapi + 2*np.sqrt(6)*kappad + 2*np.sqrt(6)*kappau - np.sqrt(6)) + np.sqrt(6)*(CoefC*(kappad + kappau + 2) + 4*CoefD)))*np.cos(theta_S))*(m23 - ma**2 - meta**2)/(24*(m23 + I*mf0*(Gammaf0 + I*mf0)))

    amp_tot += deltaI*((8*CoefA*thetapi - 2*np.sqrt(2)*CoefA*thetaprpi - 4*CoefC*thetapi - 5*np.sqrt(2)*CoefC*thetaprpi - 4*CoefD*thetapi - 8*np.sqrt(2)*CoefD*thetaprpi)*np.sin(theta_S) + 2*(5*np.sqrt(2)*CoefA*thetapi + 2*CoefA*thetaprpi - np.sqrt(2)*CoefC*thetapi - CoefC*thetaprpi + np.sqrt(2)*CoefD*thetapi + 4*CoefD*thetaprpi)*np.cos(theta_S))*(-(-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.sin(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.cos(theta_S))*(m12 + m23 - ma**2 - mpi0**2)*(m12 + m23 - meta**2 - mpi0**2)*UnitStep(m12 + m23 + 4*mK**2 - ma**2 - meta**2 - 2*mpi0**2)/(-144*I*Gammasigma*msigma + 144*m12 + 144*m23 - 144*ma**2 - 144*meta**2 - 288*mpi0**2 + 144*msigma**2) - deltaI*((8*CoefA*thetapi - 2*np.sqrt(2)*CoefA*thetaprpi - 4*CoefC*thetapi - 5*np.sqrt(2)*CoefC*thetaprpi - 4*CoefD*thetapi - 8*np.sqrt(2)*CoefD*thetaprpi)*np.sin(theta_S) + 2*(5*np.sqrt(2)*CoefA*thetapi + 2*CoefA*thetaprpi - np.sqrt(2)*CoefC*thetapi - CoefC*thetaprpi + np.sqrt(2)*CoefD*thetapi + 4*CoefD*thetaprpi)*np.cos(theta_S))*(-(-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.sin(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.cos(theta_S))*(m12 - ma**2 - mpi0**2)*(m12 - meta**2 - mpi0**2)*UnitStep(-m12 + 4*mK**2)/(144*(m12 + I*msigma*(Gammasigma + I*msigma))) + (m23 - 2*mpi0**2)*(2*CoefB*np.sin(theta_S) + np.sqrt(2)*(CoefA - CoefB)*np.cos(theta_S))*(2*(CoefA*(-3*np.sqrt(2)*deltaI*thetapi*thpiALP + 2*np.sqrt(2)*thetaALP + 2*thetaprALP) + 3*np.sqrt(2)*CoefB*deltaI*thetapi*thpiALP + 3*np.sqrt(2)*CoefB*thetaALP - np.sqrt(2)*CoefC*thetaALP - CoefC*thetaprALP + np.sqrt(2)*CoefD*thetaALP + 4*CoefD*thetaprALP + cG*(3*np.sqrt(2)*CoefA*deltaI*kappad*thetapi - 3*np.sqrt(2)*CoefA*deltaI*kappau*thetapi + 2*np.sqrt(3)*CoefA*kappad + 2*np.sqrt(3)*CoefA*kappau + CoefB*(-3*np.sqrt(2)*deltaI*kappad*thetapi + 3*np.sqrt(2)*deltaI*kappau*thetapi + 4*np.sqrt(3)*kappad + 4*np.sqrt(3)*kappau - 2*np.sqrt(3)) - np.sqrt(3)*CoefC*kappad - np.sqrt(3)*CoefC*kappau + 2*np.sqrt(3)*CoefD))*np.cos(theta_S) + (8*CoefA*thetaALP - 2*np.sqrt(2)*CoefA*thetaprALP - 12*CoefB*deltaI*thetapi*thpiALP - 12*CoefB*thetaALP - 4*CoefC*thetaALP - 5*np.sqrt(2)*CoefC*thetaprALP - 4*CoefD*thetaALP - 8*np.sqrt(2)*CoefD*thetaprALP + cG*(2*np.sqrt(6)*CoefA*(3*kappad + 3*kappau - 2) - 4*CoefB*(-3*deltaI*kappad*thetapi + 3*deltaI*kappau*thetapi + 2*np.sqrt(6)*kappad + 2*np.sqrt(6)*kappau - np.sqrt(6)) - np.sqrt(6)*(CoefC*(kappad + kappau + 2) + 4*CoefD)))*np.sin(theta_S))*(m23 - ma**2 - meta**2)*UnitStep(-m23 + 4*mK**2)/(24*m23 + 24*I*msigma*(Gammasigma + I*msigma))

    return amp_tot

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
    m12 = ma**2 + m3**2 -2*ma*x[:,0]
    m23 = ma**2 + m1**2 -2*kinematics[0]
    kappau = kappa[0,0]
    kappad = kappa[1,1]
    deltaI = (md-mu)/(md+mu)
    F0 = fpi/np.sqrt(2)
    cG = model['cG']*F0/fa
    aU3 = a_U3_repr(ma, model, fa, **kwargs)
    thpiALP = np.trace(np.dot(aU3, pi0))*2
    thetaa = np.trace(np.dot(aU3, eta))*2
    thetap = np.trace(np.dot(aU3, etap))*2
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    sm_angles = sm_mixingangles()
    thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
    thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap
    thetapi = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)* sm_angles[('eta', 'pi0')]
    thetaprpi = (s_eta+np.sqrt(2)*c_eta)*sm_angles[('etap', 'pi0')]
    I = 1j
    cq = cqhat(model, ma, **kwargs)*F0/fa
    cuhat = cq[0,0]
    cdhat = cq[1,1]

    amp_tot = 0j + (6*cG*mpi0**2*(kappad*(deltaI*(-thetapi + np.sqrt(6)) + np.sqrt(6)) + kappau*(deltaI*(thetapi - np.sqrt(6)) + np.sqrt(6))) - 3*deltaI*thetapi*(cdhat*(3*m23 - ma**2 - 3*meta**2) + cuhat*(-3*m23 + ma**2 + 3*meta**2) + 2*thpiALP*(-3*m23 + ma**2 + meta**2 + 2*mpi_pm**2)) + 2*mpi0**2*(-deltaI*thpiALP*(-3*thetapi + np.sqrt(6)) + 6*thetaALP + 3*np.sqrt(2)*thetaprALP))/(18*F0**2)

    amp_tot += -gTf2**2*(m23**2*(mf2**2 - 2*mpi_pm**2)*(ma**2 + meta**2 - mf2**2) + m23*(-ma**4*(mf2**2 - 2*mpi_pm**2) + ma**2*(2*meta**2*(mf2**2 - 2*mpi_pm**2) + mf2**4 + 2*mf2**2*mpi_pm**2) - meta**4*(mf2**2 - 2*mpi_pm**2) + meta**2*(mf2**4 + 2*mf2**2*mpi_pm**2) + 2*mf2**4*(-3*m12 + mpi_pm**2)) - 2*mf2**2*(3*m12**2*mf2**2 - 3*m12*mf2**2*(ma**2 + meta**2 + 2*mpi_pm**2) + 2*ma**4*mpi_pm**2 + ma**2*(meta**2*(3*mf2**2 - 4*mpi_pm**2) + mf2**2*mpi_pm**2) + mpi_pm**2*(2*meta**4 + meta**2*mf2**2 + 3*mf2**2*mpi_pm**2)))*(cG*kappad*(-3*deltaI*thetapi + np.sqrt(6)) + cG*kappau*(3*deltaI*thetapi + np.sqrt(6)) + 3*deltaI*thetapi*thpiALP + 2*thetaALP + np.sqrt(2)*thetaprALP)*UnitStep(m23 - (Gammaf2 - mf2)**2)/(144*mf2**4*(-m23 + mf2*(-I*Gammaf2 + mf2)))

    amp_tot += np.sqrt(6)*(2*CoefA + CoefC)*(6*cG*(2*CoefA*(kappad + kappau - 1) + CoefC) + np.sqrt(3)*(2*np.sqrt(2)*CoefA*thetaALP - 4*CoefA*thetaprALP + np.sqrt(2)*CoefC*thetaALP + 4*CoefC*thetaprALP))*((m12 + m23 - ma**2 - mpi_pm**2)*(m12 + m23 - meta**2 - mpi_pm**2)/(-I*Gammaa0_pm*ma0_pm + m12 + m23 - ma**2 + ma0_pm**2 - meta**2 - 2*mpi_pm**2) - (m12 - ma**2 - mpi_pm**2)*(m12 - meta**2 - mpi_pm**2)/(m12 + I*ma0_pm*(Gammaa0_pm + I*ma0_pm)))/144

    amp_tot += -(m23 - 2*mpi_pm**2)*(2*CoefB*np.cos(theta_S) + np.sqrt(2)*(-CoefA + CoefB)*np.sin(theta_S))*(2*(CoefA*(-3*np.sqrt(2)*deltaI*thetapi*thpiALP + 2*np.sqrt(2)*thetaALP + 2*thetaprALP) + 3*np.sqrt(2)*CoefB*deltaI*thetapi*thpiALP + 3*np.sqrt(2)*CoefB*thetaALP - np.sqrt(2)*CoefC*thetaALP - CoefC*thetaprALP + np.sqrt(2)*CoefD*thetaALP + 4*CoefD*thetaprALP + cG*(3*np.sqrt(2)*CoefA*deltaI*kappad*thetapi - 3*np.sqrt(2)*CoefA*deltaI*kappau*thetapi + 2*np.sqrt(3)*CoefA*kappad + 2*np.sqrt(3)*CoefA*kappau + CoefB*(-3*np.sqrt(2)*deltaI*kappad*thetapi + 3*np.sqrt(2)*deltaI*kappau*thetapi + 4*np.sqrt(3)*kappad + 4*np.sqrt(3)*kappau - 2*np.sqrt(3)) - np.sqrt(3)*CoefC*kappad - np.sqrt(3)*CoefC*kappau + 2*np.sqrt(3)*CoefD))*np.sin(theta_S) + (-8*CoefA*thetaALP + 2*np.sqrt(2)*CoefA*thetaprALP + 12*CoefB*deltaI*thetapi*thpiALP + 12*CoefB*thetaALP + 4*CoefC*thetaALP + 5*np.sqrt(2)*CoefC*thetaprALP + 4*CoefD*thetaALP + 8*np.sqrt(2)*CoefD*thetaprALP + cG*(-2*np.sqrt(6)*CoefA*(3*kappad + 3*kappau - 2) + 4*CoefB*(-3*deltaI*kappad*thetapi + 3*deltaI*kappau*thetapi + 2*np.sqrt(6)*kappad + 2*np.sqrt(6)*kappau - np.sqrt(6)) + np.sqrt(6)*(CoefC*(kappad + kappau + 2) + 4*CoefD)))*np.cos(theta_S))*(m23 - ma**2 - meta**2)/(24*(m23 + I*mf0*(Gammaf0 + I*mf0)))

    amp_tot += (m23 - 2*mpi_pm**2)*(2*CoefB*np.sin(theta_S) + np.sqrt(2)*(CoefA - CoefB)*np.cos(theta_S))*(2*(CoefA*(-3*np.sqrt(2)*deltaI*thetapi*thpiALP + 2*np.sqrt(2)*thetaALP + 2*thetaprALP) + 3*np.sqrt(2)*CoefB*deltaI*thetapi*thpiALP + 3*np.sqrt(2)*CoefB*thetaALP - np.sqrt(2)*CoefC*thetaALP - CoefC*thetaprALP + np.sqrt(2)*CoefD*thetaALP + 4*CoefD*thetaprALP + cG*(3*np.sqrt(2)*CoefA*deltaI*kappad*thetapi - 3*np.sqrt(2)*CoefA*deltaI*kappau*thetapi + 2*np.sqrt(3)*CoefA*kappad + 2*np.sqrt(3)*CoefA*kappau + CoefB*(-3*np.sqrt(2)*deltaI*kappad*thetapi + 3*np.sqrt(2)*deltaI*kappau*thetapi + 4*np.sqrt(3)*kappad + 4*np.sqrt(3)*kappau - 2*np.sqrt(3)) - np.sqrt(3)*CoefC*kappad - np.sqrt(3)*CoefC*kappau + 2*np.sqrt(3)*CoefD))*np.cos(theta_S) + (8*CoefA*thetaALP - 2*np.sqrt(2)*CoefA*thetaprALP - 12*CoefB*deltaI*thetapi*thpiALP - 12*CoefB*thetaALP - 4*CoefC*thetaALP - 5*np.sqrt(2)*CoefC*thetaprALP - 4*CoefD*thetaALP - 8*np.sqrt(2)*CoefD*thetaprALP + cG*(2*np.sqrt(6)*CoefA*(3*kappad + 3*kappau - 2) - 4*CoefB*(-3*deltaI*kappad*thetapi + 3*deltaI*kappau*thetapi + 2*np.sqrt(6)*kappad + 2*np.sqrt(6)*kappau - np.sqrt(6)) - np.sqrt(6)*(CoefC*(kappad + kappau + 2) + 4*CoefD)))*np.sin(theta_S))*(m23 - ma**2 - meta**2)*UnitStep(-m23 + 4*mK**2)/(24*m23 + 24*I*msigma*(Gammasigma + I*msigma))

    return amp_tot

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
    if c == 0:
        result, error = threebody_decay.decay3body(ampatoetapi0pi0, ma, m1, m2, m3, model, fa, **kwargs)
    else:
        result, error = threebody_decay.decay3body(ampatoetapipi, ma, m1, m2, m3, model, fa, **kwargs)
    return ffunction(ma)**2/(2*ma*s)*result, ffunction(ma)**2/(2*ma*s)*error

def decay_width_etapipi00(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetapipi(ma, meta, mpi0, mpi0, couplings, fa, 0, **kwargs)[0]

def decay_width_etapipipm(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetapipi(ma, meta, mpi_pm, mpi_pm, couplings, fa, 1, **kwargs)[0]

###########################    DECAY TO  a-> etap pi pi    ###########################
#It is assumed that Fpppp(m)=Fspp(m)=Ftpp(m)=F(m)

def ampatoetappi0pi0(ma, m1, m2, m3, model, fa, x, kinematics, **kwargs):
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta')
        #model: Coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a-> eta pi pi (without prefactor)

    citations.register_inspire('Aloni:2018vki')
    m12 = ma**2 + m3**2 -2*ma*x[:,0]
    m23 = ma**2 + m1**2 -2*kinematics[0]
    kappau = kappa[0,0]
    kappad = kappa[1,1]
    deltaI = (md-mu)/(md+mu)
    F0 = fpi/np.sqrt(2)
    cG = model['cG']*F0/fa
    aU3 = a_U3_repr(ma, model, fa, **kwargs)
    thpiALP = np.trace(np.dot(aU3, pi0))*2
    thetaa = np.trace(np.dot(aU3, eta))*2
    thetap = np.trace(np.dot(aU3, etap))*2
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    sm_angles = sm_mixingangles()
    thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
    thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap
    thetapi = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)* sm_angles[('eta', 'pi0')]
    thetaprpi = (s_eta+np.sqrt(2)*c_eta)*sm_angles[('etap', 'pi0')]
    I = 1j
    cq = cqhat(model, ma, **kwargs)*F0/fa
    cuhat = cq[0,0]
    cdhat = cq[1,1]

    amp_tot = 0j+ mpi0**2*(cG*kappad*(deltaI*(2*np.sqrt(2)*thetapi - thetaprpi + np.sqrt(3)) + np.sqrt(3)) + cG*kappau*(deltaI*(-2*np.sqrt(2)*thetapi + thetaprpi - np.sqrt(3)) + np.sqrt(3)) - 2*np.sqrt(2)*deltaI*thetapi*thpiALP + deltaI*thetaprpi*thpiALP - np.sqrt(3)*deltaI*thpiALP + np.sqrt(2)*thetaALP + thetaprALP)/(3*F0**2)

    amp_tot += -deltaI*gTf2**2*(-np.sqrt(2)*thetapi + 2*thetaprpi)*(((m23 - 2*mpi0**2)*(-metap**2*(m12 + m23 - ma**2 + 3*mf2**2 - 5*mpi0**2) + (mf2**2 - mpi0**2)*(m12 + m23 - ma**2 + 3*mf2**2 - mpi0**2)) + (2*mpi0**2*(2*metap**2 - 3*mf2**2) + (metap**2 - mf2**2 + mpi0**2)*(-m12 - m23 + ma**2 + mpi0**2))*(m12 - metap**2 - mpi0**2))*(-m23 + ma**2 + metap**2) + ((m23 - 2*mpi0**2)*(-metap**2*(m12 + m23 - ma**2 + 6*mf2**2 - 5*mpi0**2) + (-mf2**2 + mpi0**2)*(-m12 - m23 + ma**2 + mpi0**2)) + (-metap**2*(m12 + m23 - ma**2 + 3*mf2**2 - 5*mpi0**2) + (mf2**2 - mpi0**2)*(m12 + m23 - ma**2 + 3*mf2**2 - mpi0**2))*(m12 - metap**2 - mpi0**2))*(-m12 + ma**2 + mpi0**2) - (4*metap**2*mpi0**2*(metap**2 - 2*mf2**2 + mpi0**2) + (metap**2 - mf2**2 + mpi0**2)*(m12 + m23 - ma**2 - mpi0**2)**2 + (-m12 - m23 + ma**2 + mpi0**2)*(metap**4 - 3*metap**2*(mf2**2 - 2*mpi0**2) + 2*mf2**4 - 3*mf2**2*mpi0**2 + mpi0**4))*(m12 + m23 - metap**2 - mpi0**2))*(3*cG*(kappad - kappau) + deltaI*thetaALP*(2*thetapi + np.sqrt(2)*thetaprpi) + deltaI*(np.sqrt(2)*thetapi + thetaprpi)*(np.sqrt(3)*cG*(kappad + kappau) + thetaprALP) - 3*thpiALP)*UnitStep(-m12 - m23 + ma**2 + metap**2 + 2*mpi0**2 - (Gammaf2 - mf2)**2)/(432*mf2**4*(I*Gammaf2*mf2 - m12 - m23 + ma**2 + metap**2 - mf2**2 + 2*mpi0**2)) + deltaI*gTf2**2*(np.sqrt(2)*thetapi - 2*thetaprpi)*(cG*kappad*(np.sqrt(3)*deltaI*(np.sqrt(2)*thetapi + thetaprpi) + 3) + cG*kappau*(np.sqrt(3)*deltaI*(np.sqrt(2)*thetapi + thetaprpi) - 3) + deltaI*(2*thetaALP*thetapi + np.sqrt(2)*thetaALP*thetaprpi + np.sqrt(2)*thetapi*thetaprALP + thetaprALP*thetaprpi) - 3*thpiALP)*(m12**2*(ma**2 - mf2**2 + mpi0**2)*(metap**2 - mf2**2 + mpi0**2) - m12*(-6*m23*mf2**4 + ma**4*(metap**2 - mf2**2 + mpi0**2) + ma**2*(metap**4 - 2*metap**2*(mf2**2 + 2*mpi0**2) + mf2**4 + 6*mf2**2*mpi0**2 - mpi0**4) + metap**4*(-mf2**2 + mpi0**2) + metap**2*(mf2**4 + 6*mf2**2*mpi0**2 - mpi0**4) + 2*(-mf2**2*mpi0 + mpi0**3)**2) + 6*m23**2*mf2**4 - 6*m23*metap**2*mf2**4 - 6*m23*metap**2*mf2**2*mpi0**2 - 12*m23*mf2**4*mpi0**2 + 6*m23*mf2**2*mpi0**4 + ma**4*(metap**4 - metap**2*(mf2**2 + 2*mpi0**2) + 5*mf2**2*mpi0**2 + mpi0**4) + ma**2*(-6*m23*mf2**2*(-metap**2 + mf2**2 + mpi0**2) - metap**4*(mf2**2 + 2*mpi0**2) + metap**2*(mf2**4 - 8*mf2**2*mpi0**2 + 4*mpi0**4) + 7*mf2**4*mpi0**2 + mf2**2*mpi0**4 - 2*mpi0**6) + 5*metap**4*mf2**2*mpi0**2 + metap**4*mpi0**4 + 7*metap**2*mf2**4*mpi0**2 + metap**2*mf2**2*mpi0**4 - 2*metap**2*mpi0**6 + mf2**4*mpi0**4 - 2*mf2**2*mpi0**6 + mpi0**8)*UnitStep(m12 - (Gammaf2 - mf2)**2)/(432*mf2**4*(-m12 + mf2*(-I*Gammaf2 + mf2))) - gTf2**2*(m23**2*(mf2**2 - 2*mpi0**2)*(ma**2 + metap**2 - mf2**2) + m23*(-ma**4*(mf2**2 - 2*mpi0**2) + ma**2*(2*metap**2*(mf2**2 - 2*mpi0**2) + mf2**4 + 2*mf2**2*mpi0**2) - metap**4*(mf2**2 - 2*mpi0**2) + metap**2*(mf2**4 + 2*mf2**2*mpi0**2) + 2*mf2**4*(-3*m12 + mpi0**2)) - 2*mf2**2*(3*m12**2*mf2**2 - 3*m12*mf2**2*(ma**2 + metap**2 + 2*mpi0**2) + 2*ma**4*mpi0**2 + ma**2*(metap**2*(3*mf2**2 - 4*mpi0**2) + mf2**2*mpi0**2) + mpi0**2*(2*metap**4 + metap**2*mf2**2 + 3*mf2**2*mpi0**2)))*(cG*kappad*(-3*deltaI*thetaprpi + np.sqrt(3)) + cG*kappau*(3*deltaI*thetaprpi + np.sqrt(3)) + 3*deltaI*thetaprpi*thpiALP + np.sqrt(2)*thetaALP + thetaprALP)*UnitStep(m23 - (Gammaf2 - mf2)**2)/(144*mf2**4*(-m23 + mf2*(-I*Gammaf2 + mf2)))

    amp_tot += np.sqrt(3)*(deltaI*(m23 - 2*mpi0**2)*(2*cG*(2*CoefA*(3*deltaI*kappau*thetaprpi - 3*deltaI*thetaprpi + kappad*(3*deltaI*thetaprpi + np.sqrt(3)) - np.sqrt(3)*kappau) + CoefC*(3*deltaI*thetaprpi - 2*np.sqrt(3)*kappad + 2*np.sqrt(3)*kappau)) + np.sqrt(3)*(2*CoefA*deltaI*thetaprpi*(np.sqrt(2)*thetaALP - 2*thetaprALP) - 4*CoefA*thpiALP + CoefC*deltaI*thetaprpi*(np.sqrt(2)*thetaALP + 4*thetaprALP) + 4*CoefC*thpiALP))*(m23 - ma**2 - metap**2)*(2*np.sqrt(2)*CoefA*thetapi - 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi + 4*CoefC*thetaprpi)/(m23 + I*ma0*(Gammaa0 + I*ma0)) - 2*(CoefA - CoefC)*(cG*(2*CoefA*(kappad*(np.sqrt(6)*deltaI*thetapi - 2*np.sqrt(3)*deltaI*thetaprpi + 6) + kappau*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + 6) - 6) + CoefC*(np.sqrt(3)*deltaI*(kappad - kappau)*(np.sqrt(2)*thetapi + 4*thetaprpi) + 6)) + np.sqrt(3)*(2*CoefA*(deltaI*thpiALP*(-np.sqrt(2)*thetapi + 2*thetaprpi) + np.sqrt(2)*thetaALP - 2*thetaprALP) + CoefC*(-deltaI*thpiALP*(np.sqrt(2)*thetapi + 4*thetaprpi) + np.sqrt(2)*thetaALP + 4*thetaprALP)))*(m12 + m23 - ma**2 - mpi0**2)*(m12 + m23 - metap**2 - mpi0**2)/(-I*Gammaa0*ma0 + m12 + m23 - ma**2 + ma0**2 - metap**2 - 2*mpi0**2) + 2*(CoefA - CoefC)*(cG*(2*CoefA*(kappad*(np.sqrt(6)*deltaI*thetapi - 2*np.sqrt(3)*deltaI*thetaprpi + 6) + kappau*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + 6) - 6) + CoefC*(np.sqrt(3)*deltaI*(kappad - kappau)*(np.sqrt(2)*thetapi + 4*thetaprpi) + 6)) + np.sqrt(3)*(2*CoefA*(deltaI*thpiALP*(-np.sqrt(2)*thetapi + 2*thetaprpi) + np.sqrt(2)*thetaALP - 2*thetaprALP) + CoefC*(-deltaI*thpiALP*(np.sqrt(2)*thetapi + 4*thetaprpi) + np.sqrt(2)*thetaALP + 4*thetaprALP)))*(m12 - ma**2 - mpi0**2)*(m12 - metap**2 - mpi0**2)/(m12 + I*ma0*(Gammaa0 + I*ma0)))/72

    amp_tot += deltaI*(2*(2*CoefA*(thetapi + 2*np.sqrt(2)*thetaprpi) - CoefC*thetapi + 4*np.sqrt(2)*CoefC*thetaprpi + 4*CoefD*thetapi + 8*np.sqrt(2)*CoefD*thetaprpi)*np.sin(theta_S) + (2*np.sqrt(2)*CoefA*thetapi + 8*CoefA*thetaprpi + 5*np.sqrt(2)*CoefC*thetapi + 8*CoefC*thetaprpi + 8*np.sqrt(2)*CoefD*thetapi + 32*CoefD*thetaprpi)*np.cos(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.cos(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.sin(theta_S))*(m12 + m23 - ma**2 - mpi0**2)*(m12 + m23 - metap**2 - mpi0**2)/(-144*I*Gammaf0*mf0 + 144*m12 + 144*m23 - 144*ma**2 - 144*metap**2 + 144*mf0**2 - 288*mpi0**2) - deltaI*(2*(2*CoefA*(thetapi + 2*np.sqrt(2)*thetaprpi) - CoefC*thetapi + 4*np.sqrt(2)*CoefC*thetaprpi + 4*CoefD*thetapi + 8*np.sqrt(2)*CoefD*thetaprpi)*np.sin(theta_S) + (2*np.sqrt(2)*CoefA*thetapi + 8*CoefA*thetaprpi + 5*np.sqrt(2)*CoefC*thetapi + 8*CoefC*thetaprpi + 8*np.sqrt(2)*CoefD*thetapi + 32*CoefD*thetaprpi)*np.cos(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.cos(theta_S) + 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.sin(theta_S))*(m12 - ma**2 - mpi0**2)*(m12 - metap**2 - mpi0**2)/(144*(m12 + I*mf0*(Gammaf0 + I*mf0))) - (m23 - 2*mpi0**2)*(2*CoefB*np.cos(theta_S) + np.sqrt(2)*(-CoefA + CoefB)*np.sin(theta_S))*((2*np.sqrt(2)*CoefA*thetaALP + 8*CoefA*thetaprALP + 12*CoefB*deltaI*thetaprpi*thpiALP + 12*CoefB*thetaprALP + 5*np.sqrt(2)*CoefC*thetaALP + 8*CoefC*thetaprALP + 8*np.sqrt(2)*CoefD*thetaALP + 32*CoefD*thetaprALP + 2*cG*(2*np.sqrt(3)*CoefA - 2*CoefB*(3*deltaI*kappad*thetaprpi - 3*deltaI*kappau*thetaprpi + np.sqrt(3)*kappad + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*(2*CoefC*kappad + 2*CoefC*kappau + CoefC + 8*CoefD)))*np.cos(theta_S) + 2*(-3*np.sqrt(2)*CoefA*deltaI*thetaprpi*thpiALP + 2*CoefA*thetaALP + np.sqrt(2)*CoefA*thetaprALP + 3*np.sqrt(2)*CoefB*deltaI*thetaprpi*thpiALP + 3*np.sqrt(2)*CoefB*thetaprALP - CoefC*thetaALP + 4*np.sqrt(2)*CoefC*thetaprALP + 4*CoefD*thetaALP + 8*np.sqrt(2)*CoefD*thetaprALP + np.sqrt(2)*cG*(3*CoefA*deltaI*kappad*thetaprpi - 3*CoefA*deltaI*kappau*thetaprpi + np.sqrt(3)*CoefA*kappad + np.sqrt(3)*CoefA*kappau - CoefB*(-3*deltaI*kappau*thetaprpi + kappad*(3*deltaI*thetaprpi + np.sqrt(3)) + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*CoefC*(-2*kappad - 2*kappau + 3) + 4*np.sqrt(3)*CoefD))*np.sin(theta_S))*(m23 - ma**2 - metap**2)/(24*(m23 + I*mf0*(Gammaf0 + I*mf0)))

    amp_tot += deltaI*(-2*(2*CoefA*(thetapi + 2*np.sqrt(2)*thetaprpi) - CoefC*thetapi + 4*np.sqrt(2)*CoefC*thetaprpi + 4*CoefD*thetapi + 8*np.sqrt(2)*CoefD*thetaprpi)*np.cos(theta_S) + (2*np.sqrt(2)*CoefA*thetapi + 8*CoefA*thetaprpi + 5*np.sqrt(2)*CoefC*thetapi + 8*CoefC*thetaprpi + 8*np.sqrt(2)*CoefD*thetapi + 32*CoefD*thetaprpi)*np.sin(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.sin(theta_S) - 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.cos(theta_S))*(m12 + m23 - ma**2 - mpi0**2)*(m12 + m23 - metap**2 - mpi0**2)*UnitStep(m12 + m23 + 4*mK**2 - ma**2 - metap**2 - 2*mpi0**2)/(-144*I*Gammasigma*msigma + 144*m12 + 144*m23 - 144*ma**2 - 144*metap**2 - 288*mpi0**2 + 144*msigma**2) - deltaI*(-2*(2*CoefA*(thetapi + 2*np.sqrt(2)*thetaprpi) - CoefC*thetapi + 4*np.sqrt(2)*CoefC*thetaprpi + 4*CoefD*thetapi + 8*np.sqrt(2)*CoefD*thetaprpi)*np.cos(theta_S) + (2*np.sqrt(2)*CoefA*thetapi + 8*CoefA*thetaprpi + 5*np.sqrt(2)*CoefC*thetapi + 8*CoefC*thetaprpi + 8*np.sqrt(2)*CoefD*thetapi + 32*CoefD*thetaprpi)*np.sin(theta_S))*((-8*CoefA*deltaI*thetaALP*thetapi + 2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetaprpi + 2*np.sqrt(2)*CoefA*deltaI*thetapi*thetaprALP + 8*CoefA*deltaI*thetaprALP*thetaprpi + 4*CoefB*cG*(-np.sqrt(6)*deltaI*thetapi + 2*np.sqrt(3)*deltaI*thetaprpi + kappad*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi + 3) + kappau*(2*np.sqrt(6)*deltaI*thetapi - np.sqrt(3)*deltaI*thetaprpi - 3)) + 12*CoefB*deltaI*thetaALP*thetapi + 12*CoefB*deltaI*thetaprALP*thetaprpi - 12*CoefB*thpiALP + 4*CoefC*deltaI*thetaALP*thetapi + 5*np.sqrt(2)*CoefC*deltaI*thetaALP*thetaprpi + 5*np.sqrt(2)*CoefC*deltaI*thetapi*thetaprALP + 8*CoefC*deltaI*thetaprALP*thetaprpi + 4*CoefD*deltaI*thetaALP*thetapi + 8*np.sqrt(2)*CoefD*deltaI*thetaALP*thetaprpi + 8*np.sqrt(2)*CoefD*deltaI*thetapi*thetaprALP + 32*CoefD*deltaI*thetaprALP*thetaprpi + np.sqrt(3)*cG*deltaI*(-2*np.sqrt(2)*CoefA*thetapi*(3*kappad + 3*kappau - 2) + 4*CoefA*thetaprpi + np.sqrt(2)*CoefC*thetapi*(kappad + kappau + 2) + 2*CoefC*thetaprpi*(2*kappad + 2*kappau + 1) + 4*CoefD*(np.sqrt(2)*thetapi + 4*thetaprpi)))*np.sin(theta_S) - 2*(2*np.sqrt(2)*CoefA*deltaI*thetaALP*thetapi + 2*CoefA*deltaI*thetaALP*thetaprpi + 2*CoefA*deltaI*thetapi*thetaprALP + np.sqrt(2)*CoefA*deltaI*thetaprALP*thetaprpi + 3*np.sqrt(2)*CoefA*thpiALP + 3*np.sqrt(2)*CoefB*deltaI*thetaALP*thetapi + 3*np.sqrt(2)*CoefB*deltaI*thetaprALP*thetaprpi - 3*np.sqrt(2)*CoefB*thpiALP - np.sqrt(2)*CoefC*deltaI*thetaALP*thetapi - CoefC*deltaI*thetaALP*thetaprpi - CoefC*deltaI*thetapi*thetaprALP + 4*np.sqrt(2)*CoefC*deltaI*thetaprALP*thetaprpi + np.sqrt(2)*CoefD*deltaI*thetaALP*thetapi + 4*CoefD*deltaI*thetaALP*thetaprpi + 4*CoefD*deltaI*thetapi*thetaprALP + 8*np.sqrt(2)*CoefD*deltaI*thetaprALP*thetaprpi + cG*(CoefA*(kappad*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + kappau*(2*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi + 3*np.sqrt(2))) - CoefB*(-4*np.sqrt(3)*deltaI*kappau*thetapi + np.sqrt(6)*deltaI*kappau*thetaprpi + 2*np.sqrt(3)*deltaI*thetapi - 2*np.sqrt(6)*deltaI*thetaprpi + kappad*(-4*np.sqrt(3)*deltaI*thetapi + np.sqrt(6)*deltaI*thetaprpi - 3*np.sqrt(2)) + 3*np.sqrt(2)*kappau) + np.sqrt(3)*deltaI*(-CoefC*(kappad*thetapi + 2*np.sqrt(2)*kappad*thetaprpi + kappau*thetapi + 2*np.sqrt(2)*kappau*thetaprpi - 3*np.sqrt(2)*thetaprpi) + 2*CoefD*(thetapi + 2*np.sqrt(2)*thetaprpi))))*np.cos(theta_S))*(m12 - ma**2 - mpi0**2)*(m12 - metap**2 - mpi0**2)*UnitStep(-m12 + 4*mK**2)/(144*(m12 + I*msigma*(Gammasigma + I*msigma))) + (m23 - 2*mpi0**2)*(2*CoefB*np.sin(theta_S) + np.sqrt(2)*(CoefA - CoefB)*np.cos(theta_S))*(-(2*np.sqrt(2)*CoefA*thetaALP + 8*CoefA*thetaprALP + 12*CoefB*deltaI*thetaprpi*thpiALP + 12*CoefB*thetaprALP + 5*np.sqrt(2)*CoefC*thetaALP + 8*CoefC*thetaprALP + 8*np.sqrt(2)*CoefD*thetaALP + 32*CoefD*thetaprALP + 2*cG*(2*np.sqrt(3)*CoefA - 2*CoefB*(3*deltaI*kappad*thetaprpi - 3*deltaI*kappau*thetaprpi + np.sqrt(3)*kappad + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*(2*CoefC*kappad + 2*CoefC*kappau + CoefC + 8*CoefD)))*np.sin(theta_S) + 2*(-3*np.sqrt(2)*CoefA*deltaI*thetaprpi*thpiALP + 2*CoefA*thetaALP + np.sqrt(2)*CoefA*thetaprALP + 3*np.sqrt(2)*CoefB*deltaI*thetaprpi*thpiALP + 3*np.sqrt(2)*CoefB*thetaprALP - CoefC*thetaALP + 4*np.sqrt(2)*CoefC*thetaprALP + 4*CoefD*thetaALP + 8*np.sqrt(2)*CoefD*thetaprALP + np.sqrt(2)*cG*(3*CoefA*deltaI*kappad*thetaprpi - 3*CoefA*deltaI*kappau*thetaprpi + np.sqrt(3)*CoefA*kappad + np.sqrt(3)*CoefA*kappau - CoefB*(-3*deltaI*kappau*thetaprpi + kappad*(3*deltaI*thetaprpi + np.sqrt(3)) + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*CoefC*(-2*kappad - 2*kappau + 3) + 4*np.sqrt(3)*CoefD))*np.cos(theta_S))*(m23 - ma**2 - metap**2)*UnitStep(-m23 + 4*mK**2)/(24*m23 + 24*I*msigma*(Gammasigma + I*msigma))

    return amp_tot

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
    m12 = ma**2 + m3**2 -2*ma*x[:,0]
    m23 = ma**2 + m1**2 -2*kinematics[0]
    kappau = kappa[0,0]
    kappad = kappa[1,1]
    deltaI = (md-mu)/(md+mu)
    F0 = fpi/np.sqrt(2)
    cG = model['cG']*F0/fa
    aU3 = a_U3_repr(ma, model, fa, **kwargs)
    thpiALP = np.trace(np.dot(aU3, pi0))*2
    thetaa = np.trace(np.dot(aU3, eta))*2
    thetap = np.trace(np.dot(aU3, etap))*2
    c_eta = np.cos(theta_eta_etap)
    s_eta = np.sin(theta_eta_etap)
    sm_angles = sm_mixingangles()
    thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
    thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap
    thetapi = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)* sm_angles[('eta', 'pi0')]
    thetaprpi = (s_eta+np.sqrt(2)*c_eta)*sm_angles[('etap', 'pi0')]
    I = 1j
    cq = cqhat(model, ma, **kwargs)*F0/fa
    cuhat = cq[0,0]
    cdhat = cq[1,1]

    amp_tot = 0j+(6*cG*mpi0**2*(kappad*(deltaI*(-thetaprpi + np.sqrt(3)) + np.sqrt(3)) + kappau*(deltaI*(thetaprpi - np.sqrt(3)) + np.sqrt(3))) - 3*deltaI*thetaprpi*(cdhat*(3*m23 - ma**2 - 3*metap**2) + cuhat*(-3*m23 + ma**2 + 3*metap**2) + 2*thpiALP*(-3*m23 + ma**2 + metap**2 + 2*mpi_pm**2)) + mpi0**2*(-2*deltaI*thpiALP*(-3*thetaprpi + np.sqrt(3)) + 6*np.sqrt(2)*thetaALP + 6*thetaprALP))/(18*F0**2)

    amp_tot += -gTf2**2*(m23**2*(mf2**2 - 2*mpi_pm**2)*(ma**2 + metap**2 - mf2**2) + m23*(-ma**4*(mf2**2 - 2*mpi_pm**2) + ma**2*(2*metap**2*(mf2**2 - 2*mpi_pm**2) + mf2**4 + 2*mf2**2*mpi_pm**2) - metap**4*(mf2**2 - 2*mpi_pm**2) + metap**2*(mf2**4 + 2*mf2**2*mpi_pm**2) + 2*mf2**4*(-3*m12 + mpi_pm**2)) - 2*mf2**2*(3*m12**2*mf2**2 - 3*m12*mf2**2*(ma**2 + metap**2 + 2*mpi_pm**2) + 2*ma**4*mpi_pm**2 + ma**2*(metap**2*(3*mf2**2 - 4*mpi_pm**2) + mf2**2*mpi_pm**2) + mpi_pm**2*(2*metap**4 + metap**2*mf2**2 + 3*mf2**2*mpi_pm**2)))*(cG*kappad*(-3*deltaI*thetaprpi + np.sqrt(3)) + cG*kappau*(3*deltaI*thetaprpi + np.sqrt(3)) + 3*deltaI*thetaprpi*thpiALP + np.sqrt(2)*thetaALP + thetaprALP)*UnitStep(m23 - (Gammaf2 - mf2)**2)/(144*mf2**4*(-m23 + mf2*(-I*Gammaf2 + mf2)))

    amp_tot += np.sqrt(3)*(CoefA - CoefC)*(6*cG*(2*CoefA*(kappad + kappau - 1) + CoefC) + np.sqrt(3)*(2*np.sqrt(2)*CoefA*thetaALP - 4*CoefA*thetaprALP + np.sqrt(2)*CoefC*thetaALP + 4*CoefC*thetaprALP))*(-(m12 + m23 - ma**2 - mpi_pm**2)*(m12 + m23 - metap**2 - mpi_pm**2)/(-I*Gammaa0_pm*ma0_pm + m12 + m23 - ma**2 + ma0_pm**2 - metap**2 - 2*mpi_pm**2) + (m12 - ma**2 - mpi_pm**2)*(m12 - metap**2 - mpi_pm**2)/(m12 + I*ma0_pm*(Gammaa0_pm + I*ma0_pm)))/36

    amp_tot += -(m23 - 2*mpi_pm**2)*(2*CoefB*np.cos(theta_S) + np.sqrt(2)*(-CoefA + CoefB)*np.sin(theta_S))*((2*np.sqrt(2)*CoefA*thetaALP + 8*CoefA*thetaprALP + 12*CoefB*deltaI*thetaprpi*thpiALP + 12*CoefB*thetaprALP + 5*np.sqrt(2)*CoefC*thetaALP + 8*CoefC*thetaprALP + 8*np.sqrt(2)*CoefD*thetaALP + 32*CoefD*thetaprALP + 2*cG*(2*np.sqrt(3)*CoefA - 2*CoefB*(3*deltaI*kappad*thetaprpi - 3*deltaI*kappau*thetaprpi + np.sqrt(3)*kappad + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*(2*CoefC*kappad + 2*CoefC*kappau + CoefC + 8*CoefD)))*np.cos(theta_S) + 2*(-3*np.sqrt(2)*CoefA*deltaI*thetaprpi*thpiALP + 2*CoefA*thetaALP + np.sqrt(2)*CoefA*thetaprALP + 3*np.sqrt(2)*CoefB*deltaI*thetaprpi*thpiALP + 3*np.sqrt(2)*CoefB*thetaprALP - CoefC*thetaALP + 4*np.sqrt(2)*CoefC*thetaprALP + 4*CoefD*thetaALP + 8*np.sqrt(2)*CoefD*thetaprALP + np.sqrt(2)*cG*(3*CoefA*deltaI*kappad*thetaprpi - 3*CoefA*deltaI*kappau*thetaprpi + np.sqrt(3)*CoefA*kappad + np.sqrt(3)*CoefA*kappau - CoefB*(-3*deltaI*kappau*thetaprpi + kappad*(3*deltaI*thetaprpi + np.sqrt(3)) + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*CoefC*(-2*kappad - 2*kappau + 3) + 4*np.sqrt(3)*CoefD))*np.sin(theta_S))*(m23 - ma**2 - metap**2)/(24*(m23 + I*mf0*(Gammaf0 + I*mf0)))

    amp_tot += (m23 - 2*mpi_pm**2)*(2*CoefB*np.sin(theta_S) + np.sqrt(2)*(CoefA - CoefB)*np.cos(theta_S))*(-(2*np.sqrt(2)*CoefA*thetaALP + 8*CoefA*thetaprALP + 12*CoefB*deltaI*thetaprpi*thpiALP + 12*CoefB*thetaprALP + 5*np.sqrt(2)*CoefC*thetaALP + 8*CoefC*thetaprALP + 8*np.sqrt(2)*CoefD*thetaALP + 32*CoefD*thetaprALP + 2*cG*(2*np.sqrt(3)*CoefA - 2*CoefB*(3*deltaI*kappad*thetaprpi - 3*deltaI*kappau*thetaprpi + np.sqrt(3)*kappad + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*(2*CoefC*kappad + 2*CoefC*kappau + CoefC + 8*CoefD)))*np.sin(theta_S) + 2*(-3*np.sqrt(2)*CoefA*deltaI*thetaprpi*thpiALP + 2*CoefA*thetaALP + np.sqrt(2)*CoefA*thetaprALP + 3*np.sqrt(2)*CoefB*deltaI*thetaprpi*thpiALP + 3*np.sqrt(2)*CoefB*thetaprALP - CoefC*thetaALP + 4*np.sqrt(2)*CoefC*thetaprALP + 4*CoefD*thetaALP + 8*np.sqrt(2)*CoefD*thetaprALP + np.sqrt(2)*cG*(3*CoefA*deltaI*kappad*thetaprpi - 3*CoefA*deltaI*kappau*thetaprpi + np.sqrt(3)*CoefA*kappad + np.sqrt(3)*CoefA*kappau - CoefB*(-3*deltaI*kappau*thetaprpi + kappad*(3*deltaI*thetaprpi + np.sqrt(3)) + np.sqrt(3)*kappau - 2*np.sqrt(3)) + np.sqrt(3)*CoefC*(-2*kappad - 2*kappau + 3) + 4*np.sqrt(3)*CoefD))*np.cos(theta_S))*(m23 - ma**2 - metap**2)*UnitStep(-m23 + 4*mK**2)/(24*m23 + 24*I*msigma*(Gammasigma + I*msigma))

    return amp_tot

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
    if c == 0:
        result, error = threebody_decay.decay3body(ampatoetappi0pi0, ma, m1, m2, m3, model, fa, **kwargs)
    else:
        result, error = threebody_decay.decay3body(ampatoetappipi, ma, m1, m2, m3, model, fa, **kwargs)
    return ffunction(ma)**2/(2*ma*s)*result, ffunction(ma)**2/(2*ma*s)*error

def decay_width_etappipi00(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetappipi(ma, metap, mpi0, mpi0, couplings, fa, 0, **kwargs)[0]

def decay_width_etappipipm(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    return atoetappipi(ma, metap, mpi_pm, mpi_pm, couplings, fa, 1, **kwargs)[0]


###########################    DECAY TO  a-> pi pi gamma    ###########################
def ampatogammapipi(ma, m1, m2, m3, model, fa, x, kinematics, **kwargs):
    #INPUT
        #ma: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particle [i=1,2,3] (in GeV) (1,2: pi, 3: eta)
        #model: Coefficients
        #x: Integration variables (m12, phi, costheta, phiast, costhetaast)
        #kinematics: Kinematical relationships
    #OUTPUT
        #Amplitude a-> eta pi pi (without prefactor)
    citations.register_inspire('Aloni:2018vki')
    if ma > m1+m2+m3:
        m12 = ma**2 + m3**2 -2*ma*x[:,0]
        m23 = ma**2 + m1**2 -2*kinematics[0]
        kappau = kappa[0,0]
        kappad = kappa[1,1]
        deltaI = (md-mu)/(md+mu)
        F0 = fpi/np.sqrt(2)
        cG = model['cG']*F0/fa
        aU3 = a_U3_repr(ma, model, fa, **kwargs)
        thpiALP = np.trace(np.dot(aU3, pi0))*2
        thetaa = np.trace(np.dot(aU3, eta))*2
        thetap = np.trace(np.dot(aU3, etap))*2
        c_eta = np.cos(theta_eta_etap)
        s_eta = np.sin(theta_eta_etap)
        thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
        thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap

        integrand = -mrho_pm**4*(m12**2*m23 + m12*(m23**2 - m23*(ma**2 + 2*mpi_pm**2) - ma**2*mpi_pm**2 + mpi_pm**4) + ma**4*mpi_pm**2)*(4*cG**2*(kappad + 2*kappau)**2 + 4*cG*(kappad + 2*kappau)*(np.sqrt(6)*thetaALP + np.sqrt(3)*thetaprALP + thpiALP) + 6*thetaALP**2 + 6*np.sqrt(2)*thetaALP*thetaprALP + 2*np.sqrt(6)*thetaALP*thpiALP + 3*thetaprALP**2 + 2*np.sqrt(3)*thetaprALP*thpiALP + thpiALP**2)/(64*np.pi**3*F0**6*(Gammarho**2*mrho**2 + m12**2 - 2*m12*mrho**2 + mrho**4))
    else:
        integrand = 0.0
    return np.sqrt(integrand+0j)


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
    
    if ma > 2*mpi_pm:
        result, error = threebody_decay.decay3body(ampatogammapipi, ma, 0, mpi_pm, mpi_pm, couplings, fa, **kwargs)
        decayrate = alphaem(ma)*ffunction(ma)**2* result
        edecayrate = alphaem(ma)*ffunction(ma)**2* error
    else: decayrate, edecayrate= [0.0,0.0]
    return decayrate


def decay_width_2w(ma: float, couplings: ALPcouplings, fa: float, **kwargs):
    citations.register_inspire('Aloni:2018vki')
    if ma > 2*momega:
        kappau = kappa[0,0]
        kappad = kappa[1,1]
        F0 = fpi/np.sqrt(2)
        cG = couplings['cG']*F0/fa
        aU3 = a_U3_repr(ma, couplings, fa, **kwargs)
        thetaa = np.trace(np.dot(aU3, eta))*2
        thetap = np.trace(np.dot(aU3, etap))*2
        c_eta = np.cos(theta_eta_etap)
        s_eta = np.sin(theta_eta_etap)
        thetaALP = (c_eta-np.sqrt(2)*s_eta)/np.sqrt(2)*thetaa
        thetaprALP = (s_eta+np.sqrt(2)*c_eta)*thetap
        ampsqr = mrho_pm**4*(ma**4 - 4*ma**2*momega**2)*(3*cG*(kappad + kappau) + np.sqrt(3)*(np.sqrt(2)*thetaALP + thetaprALP))**2/(512*np.pi**4*F0**6)
        decayrate = np.abs(ampsqr)*threebody_decay.kallen(ma, momega, momega)/(16*np.pi*ma**3)*ffunction(ma)**2
    else: decayrate= 0.0
    return decayrate
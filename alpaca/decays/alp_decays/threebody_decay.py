#Program that computes the 3 body decay rate. Based on work by J. C. Romao

import numpy as np
import vegas as vegas
import math
import functools
from ...biblio.biblio import citations

#NOTES
#The input amplitude should have space for all possible kinematic variables
#Decay rates are not Lorentz invariant, which means that they do depend on frame---> Sensible idea: Get CM frame of decaying particle
#Expression I use extracted from "Integration of three body phase space: The 3BodyXSections and 3BodyDecays packages", from Jorge C. Romao

#PROCESS
# P-> q1 q2 q3 
# In CM frame of P
def kinematics(M,m1,m2,m3,Ener3,theta, thetaast, phiast):   #M,m1,m2,m3,m12,cthetaast):
    #INPUT
        #M: Mass of decaying particle
        #mi: Mass of daughter particle
        #m12: Mass of system 1-2

    #OUTPUT
        #ener1: Energy of particle 1 in reference frame of decaying particle
        #p0pi: p0.pi for i=1,2,3
        #p1pj: p0.pj for j=2,3
        #p2p3: p2.p3
        #pkpk: pk.pk = mk^2 for k=0,1,2,3

    m12 = np.sqrt(M**2+m3**2-2*M*Ener3)
    q12CM = kallen(m12,m1,m2)/(2*m12)
    Ener12 = (M**2+m12**2-m3**2)/(2*M)

    #Momenta of 1 and 2 in CM 1-2
    q1ast0 = (m12**2+m1**2-m2**2)/(2*m12)
    q1ast1 = q12CM* np.sin(thetaast)*np.cos(phiast)
    q1ast2 = q12CM* np.sin(thetaast)*np.sin(phiast)
    q1ast3 = q12CM* np.cos(thetaast)

    q2ast0 = (m12**2-m1**2+m2**2)/(2*m12)
    q2ast1 = -q12CM* np.sin(thetaast)*np.cos(phiast)
    q2ast2 = -q12CM* np.sin(thetaast)*np.sin(phiast)
    q2ast3 = -q12CM* np.cos(thetaast)


    q10 = Ener12/m12*(q1ast0+np.sqrt(1-m12**2/Ener12**2)*q1ast3)
    q11 = -q1ast1*np.cos(theta)-Ener12/m12*np.sin(theta)*(q1ast0*np.sqrt(1-m12**2/Ener12**2)+q1ast3)
    q12 = q1ast2
    q13 = q1ast1*np.sin(theta)-Ener12/m12*np.cos(theta)*(np.sqrt(1-m12**2/Ener12**2)*q1ast0+q1ast3)

    q20 = Ener12/m12*(q2ast0+np.sqrt(1-m12**2/Ener12**2)*q2ast3)
    q21 = -q2ast1*np.cos(theta)-Ener12/m12*np.sin(theta)*(q2ast0*np.sqrt(1-m12**2/Ener12**2)+q2ast3)
    q22 = q2ast2
    q23 = q2ast1*np.sin(theta)-Ener12/m12*np.cos(theta)*(np.sqrt(1-m12**2/Ener12**2)*q2ast0+q2ast3)

    
    p0p0 = M**2
    q1q1 = m1**2
    q2q2 = m2**2
    q3q3 = m3**2

    p0q1 = M*q10
    p0q2 = M*q20
    p0q3 = (M**2+m3**2-m12**2)/2
    
    q1q2 = q10*q20-(q11*q21+q12*q22+q13*q23)
    q1q3 = p0q1-q1q1-q1q2
    q2q3 = p0q2-q2q2-q1q2
    
    return [p0q1,p0q2,p0q3,q1q2,q1q3,q2q3,p0p0,q1q1,q2q2,q3q3]

def pCM(x, y, z):
    #INPUT:
        #x, y, z: Mass (in GeV)
    #OUTPUT:
        #p: Momentum in CM frame of particle with mass x
    return np.sqrt((x**2-(y+z)**2)*(x**2-(y-z)**2))/(2*x)

def integrand(amplitude, M, m1, m2, m3, model, fa, x, **kwargs):
    #INPUT:
        #amplitude: Amplitude expression
        #M: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particles [i=1,2,3] (in GeV)
        #fa: Scale of U(1)
        #x: Integration variables (for vegas)
    #OUTPUT:
        #Integrand of decay rate

    Ener3 = x[:,0] #Energy of particle 3 in frame decaying particle
    theta = x[:,1] #Angle theta, CM frame of decaying particle
    thetaast = x[:,2] #Angle theta, CM frame of system 1-2
    phiast = x[:,3] #phi angle, CM frame of system 1-2
    m12 = np.sqrt(M**2+m3**2-2*M*Ener3)

    ampl = amplitude(M, m1, m2, m3, model, fa, x, kinematics(M,m1,m2,m3,Ener3,theta, thetaast, phiast), **kwargs)

    return np.abs(ampl)**2 * np.sqrt(Ener3**2-m3**2)* kallen(m12, m1,m2)/m12**2 *np.sin(theta) *np.sin(thetaast)

def integrand_spheric(amplitude, M, m1, m2, m3, model, fa, Ener3, **kwargs):
    #INPUT:
        #amplitude: Amplitude expression
        #M: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particles [i=1,2,3] (in GeV)
        #fa: Scale of U(1)
        #x: Integration variables (for vegas)
    #OUTPUT:
        #Integrand of decay rate

    m12 = np.sqrt(M**2+m3**2-2*M*Ener3)

    ampl = amplitude(M, model, fa, Ener3, **kwargs)

    return np.abs(ampl)**2 * np.sqrt(Ener3**2-m3**2)* kallen(m12, m1,m2)/m12**2

def kallen(x,y,z):
    #INPUT:
        #x, y, z: Energy or mass (in GeV)
    #OUTPUT:
        #p: Lambda function
    return np.sqrt(x**4+y**4+z**4-2*pow(x*y,2)-2*pow(x*z,2)-2*pow(y*z,2))


def decay3body(amplitude, M, m1, m2, m3, model, fa, **kwargs):
    #INPUT:
        #amplitude: Amplitude expression
        #M: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particles [i=1,2,3] (in GeV)
    #OUTPUT:
        #Decay rate
    citations.register_bibtex('romao', "@book{Romao, title={Integration of three body phase space: The 3BodyXSections and 3BodyDecays packages}, url={https://porthos.tecnico.ulisboa.pt/CTQFT/files/ThreeBodyPhaseSpace.pdf}, author={Romao, Jorge C.} }")
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
    nitn_adapt = kwargs.get('nitn_adapt', 10)
    neval_adapt = kwargs.get('neval_adapt', 10)
    nitn = kwargs.get('nitn', 10)
    neval = kwargs.get('neval', 100)
    cores = kwargs.get('cores', 1)
    kwargs_integrand = {k: v for k, v in kwargs.items() if k not in ['nitn_adapt', 'neval_adapt', 'nitn', 'neval', 'cores']}
    q3max=kallen(M,m1+m2,m3)/(2*M)
    E3max=np.sqrt(q3max**2+m3**2)
    integrator= vegas.Integrator([[m3,E3max],[0, np.pi], [0, np.pi],[0, 2*np.pi]], nproc=cores)#[-1, 1], [-1, 1],[0, 2*np.pi]]) #vegas.Integrator([[(m1+m2)**2,(M-m3)**2],[0, 2*np.pi], [-1, 1],[0, 2*np.pi],[-1, 1]])
    # step 1 -- adapt to integrand; discard results
    integrator(vegas.lbatchintegrand(functools.partial(integrand, amplitude, M, m1, m2, m3, model, fa, **kwargs_integrand)), nitn=nitn_adapt, neval=neval_adapt)
    # step 2 -- integrator has adapted to integrand; keep results
    resint = integrator(vegas.lbatchintegrand(functools.partial(integrand, amplitude, M, m1, m2, m3, model, fa, **kwargs_integrand)), nitn=nitn, neval=neval)
    decayrate = 1/((2*np.pi)**4*(32*M))* resint.mean
    edecayrate = 1/((2*np.pi)**4*(32*M))* resint.sdev
    return decayrate, edecayrate


def decay3body_spheric(amplitude, M, m1, m2, m3, model, fa, **kwargs):
    #INPUT:
        #amplitude: Amplitude expression
        #M: Mass of decaying particle (in GeV)
        #mi: Mass of daughter particles [i=1,2,3] (in GeV)
    #OUTPUT:
        #Decay rate
    citations.register_bibtex('romao', "@book{Romao, title={Integration of three body phase space: The 3BodyXSections and 3BodyDecays packages}, url={https://porthos.tecnico.ulisboa.pt/CTQFT/files/ThreeBodyPhaseSpace.pdf}, author={Romao, Jorge C.} }")
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
    nitn_adapt = kwargs.get('nitn_adapt', 10)
    neval_adapt = kwargs.get('neval_adapt', 10)
    nitn = kwargs.get('nitn', 10)
    neval = kwargs.get('neval', 100)
    cores = kwargs.get('cores', 1)
    kwargs_integrand = {k: v for k, v in kwargs.items() if k not in ['nitn_adapt', 'neval_adapt', 'nitn', 'neval', 'cores']}
    q3max=kallen(M,m1+m2,m3)/(2*M)
    E3max=np.sqrt(q3max**2+m3**2)
    integrator= vegas.Integrator([m3,E3max], nproc=cores)#[-1, 1], [-1, 1],[0, 2*np.pi]]) #vegas.Integrator([[(m1+m2)**2,(M-m3)**2],[0, 2*np.pi], [-1, 1],[0, 2*np.pi],[-1, 1]])
    # step 1 -- adapt to integrand; discard results
    integrator(vegas.lbatchintegrand(functools.partial(integrand_spheric, amplitude, M, m1, m2, m3, model, fa, **kwargs_integrand)), nitn=nitn_adapt, neval=neval_adapt)
    # step 2 -- integrator has adapted to integrand; keep results
    resint = integrator(vegas.lbatchintegrand(functools.partial(integrand_spheric, amplitude, M, m1, m2, m3, model, fa, **kwargs_integrand)), nitn=nitn, neval=neval)
    decayrate = 1/((2*np.pi)**4*(32*M))* resint.mean*8*np.pi
    edecayrate = 1/((2*np.pi)**4*(32*M))* resint.sdev*8*np.pi
    return decayrate, edecayrate

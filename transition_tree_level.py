'''Amplitude of the processes M1 -> M2 a at tree level, as obtained in 2211.08343'''
import numpy as np
#from ...rge import ALPcouplings, bases_above
#from ...constants import GF, mc
#from ...biblio.biblio import citations
import scipy.integrate as integrate
GF=1.16e-5
mc=1.27
#k.PI=(MI**2-MF**2+ma**2)/2
#k.PF=(MI**2-MF**2-ma**2)/2
#PI.PF=(MI**2+MF**2-ma**2)/2


def deltaaM(MI, ma):
    return ma/(2*MI)

def phi(x:float, M:float, mQ:float, mq:float):
    if M > mc:
        xi = 1/(1+mQ/mq) 
        fun = pow(xi**2/(1-x)+1/x-1,-2)
    else:
        fun = x*(1-x)
    return fun

def g(M:float):
    if M > mc:
        fun = 1
    else:
        fun = 0.01
    return fun

#ma: float, couplings: ALPcouplings, fa: float, **kwargs
# Pseudoscalar to pseudoscalar
## s-channel
def pseudo_to_pseudo_schannel_initial(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mq:float, mQ:float, cq:float, cQ:float, VCKM:float, *args):
    #citations.register_inspire('Guerrera:2022ykl')
    kPI = (MI**2-MF**2+ma**2)/2
    kPF = (MI**2-MF**2-ma**2)/2
    integral = integrate.quad(lambda x: g(MI)*phi(x, MI, mQ, mq)*(cq*mq*np.heaviside(1-x, deltaaM(MI,ma))/(ma**2 - 2* kPI*(1-x)) - cQ*mQ *np.heaviside(x, deltaaM(MI,ma))/(ma**2 - 2* kPI*x)), 0, 1-1e-5)
    return GF*VCKM*fI*fF/(np.sqrt(2)*fa)*MI*kPF*integral[0]

def pseudo_to_pseudo_schannel_final(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mqp:float, mQp:float, cqp:float, cQp:float, VCKM:float, *args):
    #citations.register_inspire('Guerrera:2022ykl')
    kPI = (MI**2-MF**2+ma**2)/2
    kPF = (MI**2-MF**2-ma**2)/2
    integral = integrate.quad(lambda y: g(MF)*phi(y, MF, mQp, mqp)*(cQp*mQp/(ma**2 + kPF*y) - cqp*mqp/(ma**2 + 2* kPF*(1-y))), 0, 1-1e-5)
    return GF*VCKM*fI*fF/(np.sqrt(2)*fa)*MF*kPI*integral[0]

## t-channel
def pseudo_to_pseudo_tchannel_initial(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mq:float, mQ:float, cq:float, cQ:float, VCKM:float, *args):
    #citations.register_inspire('Guerrera:2022ykl')
    kPI = (MI**2-MF**2+ma**2)/2
    kPF = (MI**2-MF**2-ma**2)/2
    integral = integrate.quad(lambda x: g(MI)*phi(x, MI, mQ, mq)*(cQ*mQ*np.heaviside(x, deltaaM(MI,ma))/(ma**2 - 2* kPI*x) - cq*mq *np.heaviside(1-x, deltaaM(MI,ma))/(ma**2 - 2* kPI*(1-x))), 0, 1-1e-5)
    return GF*VCKM*fI*fF/(2*fa)*MI*kPF*integral[0]

def pseudo_to_pseudo_tchannel_final(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mqp:float, mQp:float, cqp:float, cQp:float, VCKM:float, *args):
    #citations.register_inspire('Guerrera:2022ykl')
    kPI = (MI**2-MF**2+ma**2)/2
    kPF = (MI**2-MF**2-ma**2)/2
    integral = integrate.quad(lambda y: g(MF)*phi(y, MF, mQp, mqp)*(cqp*mqp/(ma**2 + 2* kPF*(1-y)) - cQp*mQp/(ma**2 + 2* kPF*y)), 0, 1-1e-5)
    return GF*VCKM*fI*fF/(2*fa)*MF*kPI*integral[0]

# Pseudoscalar to vector
## s-channel
def pseudo_to_vector_schannel_initial(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mq:float, mQ:float, cq:float, cQ:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    pIepspFnum = pIepspF(MI, MF, ma)
    kPI = (MI**2-MF**2+ma**2)/2
    integral = integrate.quad(lambda x: g(MI)*phi(x, MI, mQ, mq)*(cq*mq*np.heaviside(1-x, deltaaM(MI,ma))/(ma**2 - kPI*(1-x)) - cQ*mQ *np.heaviside(x, deltaaM(MI,ma))/(ma**2 - 2* kPI*x)), 0, 1-1e-5)
    return 1j*GF*VCKM*fI*fF/(np.sqrt(2)*fa)*MI*MF*np.sum(pIepspFnum)*integral[0]

def pseudo_to_vector_schannel_final(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mqp:float, mQp:float, cqp:float, cQp:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    pIepspFnum = pIepspF(MI, MF, ma)
    kPF = (MI**2-MF**2-ma**2)/2
    integral = integrate.quad(lambda y: g(MF)*phi(y, MF, mQp, mqp)*(cQp*mQp/(ma**2 + 2* kPF*y) - cqp*mqp/(ma**2 + 2* kPF*(1-y))), 0, 1-1e-5)
    return 1j*GF*VCKM*fI*fF/(np.sqrt(2)*fa)*(-MF**2*np.sum(pIepspFnum))*integral[0]

## t-channel
def pseudo_to_vector_tchannel_initial(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mq:float, mQ:float, cq:float, cQ:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    pIepspFnum = pIepspF(MI, MF, ma)
    kPI = (MI**2-MF**2+ma**2)/2
    integral = integrate.quad(lambda x: g(MI)*phi(x, MI, mQ, mq)*(cQ*mQ*np.heaviside(x, deltaaM(MI,ma))/(ma**2 - kPI*x) - cq*mq *np.heaviside(1-x, deltaaM(MI,ma))/(ma**2 - 2* kPI*(1-x))), 0, 1-1e-5)
    return 1j*GF*VCKM*fI*fF/(2*fa)*MI*MF*np.sum(pIepspFnum)*integral[0]

def pseudo_to_vector_tchannel_final(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mqp:float, mQp:float, cqp:float, cQp:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    pIepspFnum = pIepspF(MI, MF, ma)
    kPF = (MI**2-MF**2-ma**2)/2
    integral = integrate.quad(lambda y: phi(y, MF, mQp, mqp)*(cQp*mQp/(ma**2 + 2* kPF*y) - cqp*mqp/(ma**2 + 2* kPF*(1-y))), 0, 1-1e-5)
    return 1j*GF*VCKM*fI*fF/(2*fa)*(-MF**2*np.sum(pIepspFnum))*integral[0]


# Vector to pseudoscalar
## s-channel
def vector_to_pseudo_schannel_initial(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mq:float, mQ:float, cq:float, cQ:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    kPI = (MI**2-MF**2+ma**2)/2
    pFepspInum = pFepspI(MI, MF, ma)
    integral = integrate.quad(lambda x: phi(x, MI, mQ,mq)*((cq*mq*np.heaviside(1-x, deltaaM(MI,ma))))/(ma**2-2* kPI*(1-x))-(cQ*mQ*np.heaviside(x,deltaaM(MI,ma)))/(ma**2-2*kPI*x), 0, 1-1e-5)
    return -1j*GF*VCKM*fI*fF/(np.sqrt(2)*fa)*(-MI**2*np.sum(pFepspInum))*integral[0]

def vector_to_pseudo_schannel_final(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mqp:float, mQp:float, cqp:float, cQp:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    kPF = (MI**2-MF**2-ma**2)/2
    pIepspFnum = pIepspF(MI, MF, ma)
    integral = integrate.quad(lambda y: phi(y, MI, mQp, mqp)*((cQp*mQp)/(ma**2+2*kPF*y)-(cqp*mqp)/(ma**2+2*kPF*(1-y))), 0, 1-1e-5)
    return -1j*GF*VCKM*fI*fF/(np.sqrt(2)*fa)*MI*MF*(np.sum(pIepspFnum))*integral[0]
## t-channel
def vector_to_pseudo_tchannel_initial(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mq:float, mQ:float, cq:float, cQ:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    kPI = (MI**2-MF**2+ma**2)/2
    pFepspInum = pFepspI(MI, MF, ma)
    integral = integrate.quad(lambda x: phi(x, MI, mQ, mq)*((cQ*mQ*np.heaviside(x, deltaaM(MI,ma))))/(ma**2-2* kPI*x)-(cq*mq*np.heaviside(1-x,deltaaM(MI,ma)))/(ma**2-2*kPI*(1-x)), 0, 1-1e-5)
    return -1j*GF*VCKM*fI*fF/(2*fa)*(-MI**2*np.sum(pFepspInum))*integral[0]

def vector_to_pseudo_tchannel_final(MI:float, MF:float, ma:float, fI: float, fF: float, fa:float, mqp:float, mQp:float, cqp:float, cQp:float, VCKM:float):
    #citations.register_inspire('Guerrera:2022ykl')
    kPF = (MI**2-MF**2-ma**2)/2
    pIepspFnum = pIepspF(MI, MF, ma)
    integral = integrate.quad(lambda y: g(MF)*phi(y, MI, mQp, mqp)*((cQp*mQp)/(ma**2+2*kPF*y)-(cqp*mqp)/(ma**2+2*kPF*(1-y))), 0, 1-1e-5)
    return -1j*GF*VCKM*fI*fF/(2*fa)*MI*MF*integral[0]*np.sum(pIepspFnum)

def pCM(MI:float, MF:float, ma:float):
    return np.sqrt((MI**2-(MF+ma)**2)*(MI**2-(MF-ma)**2))/(2*MI)

def pIepspF(MI:float, MF:float, ma:float):
    p3F = pCM(MI, MF, ma)
    return 0, 0, MI/MF*p3F

def pFepspI(MI:float, MF:float, ma:float):
    p3F = pCM(MI, MF, ma)
    return 0, 0, -p3F


def amp_squared(amp_initial, amp_final, MI:float, MF:float, ma:float, fI:float, fF:float, fa:float, mq:float, mQ:float, cq:float, cQ:float, VCKM:float):
    #Do loop on polarization vectors--> 2 loops, one for amp and 1 for amp conj
    return abs(amp_initial(MI, MF, ma, fI, fF, fa, mq, mQ, cq, cQ, VCKM)+ amp_final(MI, MF, ma, fI, fF, fa, mq, mQ, cq, cQ, VCKM))


mB=5
mK=0.5
ma=1
fB=0.1
fK=0.1
fa=1000
mu=2e-3
ms=90e-3
Vus=0.1
Vub=0.001
print(amp_squared(vector_to_pseudo_tchannel_initial,vector_to_pseudo_tchannel_final,mB,mK, ma, fB, fK, fa, mu, ms, 1,1, Vus*Vub))

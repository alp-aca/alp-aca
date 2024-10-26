import numpy as np 
import matplotlib.pyplot as plt
# from alpaca.decays.alp_decays.fermion_decays import decay_width_electron, decay_width_muon, decay_width_tau, decay_width_charm, decay_width_bottom
# from alpaca.decays.alp_decays.hadronic_decays_def import decay_width_3pi000, decay_width_3pi0pm, decay_width_etapipi00, decay_width_etapipipm, decay_width_gammapipi, decay_width_gluongluon
# from alpaca.decays.alp_decays.gaugebosons import decay_width_2gamma
from alpaca.rge import ALPcouplings
from alpaca.decays.alp_decays.branching_ratios import BR_2photons, BR_electron, BR_muon, BR_tau, BR_charm, BR_bottom, BR_3pis, BR_etapipi, BR_gammapipi, BR_gluongluon

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})
plt.rcParams.update({'font.size': 14})

#ma=np.logspace(-1,1,1000)
ma=np.linspace(0.1,10,100)
fa=1000

cphoton=1.0
cfermi=1.0
cgluons=1.0
#low_scale=2.0
# couplings=ALPcouplings({'cgamma': cphoton, 'cuA': cfermi, 'cdA': cfermi, 'ceA': cfermi}, scale=, basis='VA_below')


BR_2gamma=[]
BRelec=[]
BRmu=[]
BRtau=[]
BRcharm=[]
BRbottom=[]
BR3pis=[]
BRetapipi=[]
BRgammapipi=[]
BRgluongluon=[]

for m in ma:
    couplings = ALPcouplings({'cgamma': cphoton, 'cuA': cfermi, 'cdA': cfermi, 'ceA': cfermi}, scale=m, basis='VA_below')
    BR_2gamma.append(BR_2photons(m, fa, couplings))
    BRelec.append(BR_electron(m, fa, couplings))
    BRmu.append(BR_muon(m, fa, couplings))
    BRtau.append(BR_tau(m, fa, couplings))
    BRcharm.append(BR_charm(m, fa, couplings))
    BRbottom.append(BR_bottom(m, fa, couplings))
    BR3pis.append(BR_3pis(m, fa, couplings))
    BRetapipi.append(BR_etapipi(m, fa, couplings))
    BRgammapipi.append(BR_gammapipi(m, fa, couplings))
    BRgluongluon.append(BR_gluongluon(m, fa, couplings))


    

fig, ax = plt.subplots() 

ax.plot(ma, BR_2gamma, label=r"$\textrm{BR}\left(a\rightarrow \gamma\gamma\right)$")
ax.plot(ma, BRelec, label=r"$\textrm{BR}\left(a\rightarrow e^+e^-\right)$")
ax.plot(ma, BRmu, label=r"$\textrm{BR}\left(a\rightarrow \mu^+\mu^-\right)$")
ax.plot(ma, BRtau, label=r"$\textrm{BR}\left(a\rightarrow \tau^+\tau^-\right)$")
ax.plot(ma, BRcharm, label=r"$\textrm{BR}\left(a\rightarrow c\bar{c}\right)$")
ax.plot(ma, BRbottom, label=r"$\textrm{BR}\left(a\rightarrow b\bar{b}\right)$")
ax.plot(ma, BR3pis, label=r"$\textrm{BR}\left(a\rightarrow 3\pi^0\right)$")
ax.plot(ma, BRetapipi, label=r"$\textrm{BR}\left(a\rightarrow \eta\pi\pi\right)$")
ax.plot(ma, BRgammapipi, label=r"$\textrm{BR}\left(a\rightarrow \gamma\pi\pi\right)$")



ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_a \, \left[\textrm{GeV}\right]$")
ax.set_ylabel(r"$\Gamma_a \,\left[\textrm{GeV}\right]$")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

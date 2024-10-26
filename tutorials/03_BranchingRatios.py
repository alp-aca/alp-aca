import numpy as np 
import matplotlib.pyplot as plt
from alpaca.decays.alp_decays.fermion_decays import decay_width_electron, decay_width_muon, decay_width_tau, decay_width_charm, decay_width_bottom
from alpaca.decays.alp_decays.hadronic_decays_def import decay_width_3pi000, decay_width_3pi0pm, decay_width_etapipi00, decay_width_etapipipm, decay_width_gammapipi, decay_width_gluongluon
from alpaca.decays.alp_decays.gaugebosons import decay_width_2gamma
from alpaca.rge import ALPcouplings


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})
plt.rcParams.update({'font.size': 14})

ma=np.logspace(-1,1,1000)
fa=1000

cphoton=1.0
cfermi=1.0
cgluons=1.0
low_scale=2.0
# couplings=ALPcouplings({'cgamma': cphoton, 'cuA': cfermi, 'cdA': cfermi, 'ceA': cfermi}, scale=, basis='VA_below')

DW_elec=[]
DW_muon=[]
DW_tau=[]
DW_charm=[]
DW_bottom=[]
DW_3pis=[]
DW_etapipi=[]
DW_gammapipi=[]
DW_gluongluon=[]
DW_2photons=[]
for m in ma:
    couplings=ALPcouplings({'cgamma': cphoton, 'cg': cgluons, 'cuA': cfermi, 'cdA': cfermi, 'ceA': cfermi}, scale=m, basis='VA_below')
    DW_elec.append(decay_width_electron(m, couplings, fa))
    DW_muon.append(decay_width_muon(m, couplings, fa))
    DW_tau.append(decay_width_tau(m, couplings, fa))
    DW_charm.append(decay_width_charm(m, couplings, fa))
    DW_bottom.append(decay_width_bottom(m, couplings, fa))
    DW_3pis.append(decay_width_3pi000(m, couplings, fa))
    DW_etapipi.append(decay_width_etapipi00(m, couplings, fa))
    DW_gammapipi.append(decay_width_gammapipi(m, couplings, fa))
    DW_gluongluon.append(decay_width_gluongluon(m, fa))
    DW_2photons.append(decay_width_2gamma(m, couplings, fa))

    

fig, ax = plt.subplots() 

ax.plot(ma, DW_elec, label=r"$e e$")
ax.plot(ma, DW_muon, label=r"$\mu\mu$")
ax.plot(ma, DW_tau, label=r"$\tau\tau$")
ax.plot(ma, DW_charm, label=r"$cc$")
ax.plot(ma, DW_bottom, label=r"$b b$")
ax.plot(ma, DW_3pis, label=r"$3\pi$")
ax.plot(ma, DW_etapipi, label=r"$\eta\pi\pi$")
ax.plot(ma, DW_gammapipi, label=r"$\gamma\pi\pi$")
ax.plot(ma, DW_gluongluon, label=r"$gg$")
ax.plot(ma, DW_2photons, label=r"$\gamma\gamma$")



ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_a \, \left[\textrm{GeV}\right]$")
ax.set_ylabel(r"$\Gamma_a \,\left[\textrm{GeV}\right]$")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from alpaca import ALPcouplings
from alpaca.decays.alp_decays.branching_ratios import total_decay_width
from multiprocessing import Pool
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
import seaborn as sns
import time

start_time = time.time()


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})
plt.rcParams.update({'font.size': 14})

speed_of_light = 299792458  # m/s

def kallen_funcition(x, y, z):
    return x**2 + y**2 + z**2 - 2*x*y - 2*x*z - 2*y*z

def gamma_boost_a(mB, ma, mX, gammaexp, gammabeta, theta):
    return (gammaexp * (mB**2 + ma**2 - mX**2) + gammabeta * np.sqrt(kallen_funcition(mB**2, ma**2, mX**2)) * np.cos(theta)) / (2 * mB * ma)

def gamma_beta_a(mB, ma, mX, gammaexp, gammabeta, theta):
    return np.sqrt(gamma_boost_a(mB, ma, mX, gammaexp, gammabeta, theta)**2 - 1)

mB = 5.279
mX = 0.493
cphoton = 1.0
clept = 1.0
cu = 1.0
cd = 1.0 
cgluons = 1.0

ma_vec = np.linspace(0.1, mB, 100)
fa_vec = np.logspace(0, 6, 50)

gamma_beta_belle2 = 0.28
gamma_belle2 = np.sqrt(1 + gamma_beta_belle2**2)

def compute_dw(m):
    couplings = ALPcouplings({'cgamma': cphoton, 'cuA': cu, 'cdA': cd, 'ceA': clept}, scale=m, basis='VA_below')
    return [total_decay_width(m, couplings, f)['DW_tot'] for f in fa_vec]

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        DWs = pool.map(compute_dw, ma_vec)

    DWs = np.array(DWs)
    Z = speed_of_light * 100 / (1.52 * 10**24 * DWs)

    # Ensure Z is real
    Z = np.real(Z)

    # Create the contour plot
    fig, ax = plt.subplots()
    ma_grid, fa_grid = np.meshgrid(ma_vec, fa_vec)


    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

    contour = ax.contourf(ma_grid, Z.T, fa_grid, levels=fa_vec, cmap=cmap, norm=LogNorm())

    # Add a color bar
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(r'$f_a$')
    cbar.set_ticks([10**i for i in range(0, 7)])  # Set tick locations as powers of 10 within your data range

    # Calculate the dashed curves
    curve2 = -0.1 / (np.log(0.95) * gamma_beta_a(mB, ma_vec, mX, gamma_belle2, gamma_beta_belle2, 0))
    curve1 = -100 / (np.log(0.95) * gamma_beta_a(mB, ma_vec, mX, gamma_belle2, gamma_beta_belle2, 0))

    # Plot the dashed curves
    ax.plot(ma_vec, curve1, 'k--')
    ax.plot(ma_vec, curve2, 'k--')

    ax.fill_between(ma_vec, curve1, 10**8, where=(10**8 > curve1), color='salmon', alpha=0.2)
    ax.fill_between(ma_vec, curve1, curve2, where=(curve1 > curve2), color='salmon', alpha=0.4)
    ax.fill_between(ma_vec, curve2, 0, where=(curve2 > 0), color='salmon', alpha=0.6)

    ax.set_xlim(min(ma_vec), max(ma_vec)-mX)
    ax.set_ylim(min(Z.flatten()), max(Z.flatten()))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$m_a \, \left[\textrm{GeV}\right]$')
    ax.set_ylabel(r'$\frac{c}{\Gamma} \, \left[\textrm{cm}\right]$')
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    plt.show()

# fig, ax = plt.subplots()

# ax.plot(ma_vec, speed_of_light*100/(1.52*10**24*np.array(DWs)))

# ax.plot(ma_vec, -0.1/(np.log(0.95)*gamma_beta_a(mB,ma_vec, mX,gamma_belle2, gamma_beta_belle2,0)), 'k--')
# ax.plot(ma_vec, -100/(np.log(0.95)*gamma_beta_a(mB,ma_vec, mX,gamma_belle2, gamma_beta_belle2,0)), 'k--')
# # ax.plot(ma_vec, 1/(np.sqrt(boost_a(mB,ma_vec, mX,np.sqrt(1**2), 0,0)**2-1)*speed_of_light), 'r--')
# # ax.plot(ma_vec, 1/(np.sqrt(boost_a(mB,ma_vec, mX,np.sqrt(1**2), 0,np.pi)**2-1)*speed_of_light), 'r--')
# ax.hlines(0.1, min(ma_vec), max(ma_vec),'r', '--')
# ax.hlines(100, min(ma_vec), max(ma_vec),'r', '--')

# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel(r'$m_a \, \mathrm{[GeV]}$')
# ax.set_ylabel(r'$c\tau_a \, \mathrm{[cm]}$')

# plt.show()


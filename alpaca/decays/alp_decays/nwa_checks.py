import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def nwa_validity(ma, Gamma):
    '''
    Function that checks the validity of the narrow-width approximation (NWA)
    INPUT:
        ma: Mass of the ALP (GeV)
        Gamma: Decay width of ALP (GeV)
    OUTPUT:
        val: 0 (not valid) or 1 (valid)
    '''
    if Gamma < 10*ma: val = 1
    else: val = 0
    return val

def nwa_contour(X, Y, Z):
    '''
    Contour plot with levels 0 (grey) and 1 (transparent)
    INPUT:
        X: X-axis
        Y: Y-axis
        Z: Z-axis
    OUTPUT:
        contourf: Contour plot with shaded regions where NWA not valid
    '''
    colors = [(0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0)]  # Grey and transparent
    plt.contourf(X, Y, Z, levels=(0,1), cmap = ListedColormap(colors))
    return 0

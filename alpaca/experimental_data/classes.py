import numpy as np
from scipy.stats import chi2
from ..citations import citations
from ..common import kallen
from scipy.integrate import quad_vec
import pandas as pd
from scipy.interpolate import interp1d

rmin_belle = 0.1
rmax_belle = 100
rmin_belleII = 0.1
rmax_belleII = 100
rmin_babar = 0.1
rmax_babar = 100
rmin_besIII = 0.1
rmax_besIII = 100

def sigma(cl, df, param):
    #INPUT:
        #cl: Confidence level of measurement
        #df: Degrees of freedom (df)
        #param: Measured quantity
    #OUTPUT:
        #Value of standard deviation of the measurement with said confidence level
    p_value = 1 - cl
    # Calculate the chi-squared value
    chi_squared_value = chi2.ppf(1 - p_value, df)
    return param/np.sqrt(chi_squared_value)


class MeasurementBase:
    def __init__(self, inspire_id: str, type: str, rmin: float|None = None, rmax: float|None = None, lab_boost: float = 0.0, mass_parent: float = 0.0, mass_sibling: float = 0.0):
        """
        Initialize an instance of the class.

        Parameters:
        inspire_id (str): Inspire-HEP reference of the measurement.
        type (str): The type of the instance.
        rmin (float, optional): The minimum radius. Defaults to None.
        rmax (float, optional): The maximum radius. Defaults to None.
        lab_boost (float): The laboratory boost value. Defaults to 0.0.
        mass_parent (float): The mass of the parent. Defaults to 0.0.
        mass_sibling (float): The mass of the sibling. Defaults to 0.0.
        """
        self.inspire_id = inspire_id
        self.type = type
        self.rmin = rmin
        self.rmax = rmax
        self.initiated = False
        self.lab_boost = lab_boost
        self.mass_parent = mass_parent
        self.mass_sibling = mass_sibling

    def initiate(self):
        if not self.initiated:
            self.initiated = True
            citations.register_inspire(self.inspire_id)
        
    def get_central(self, ma: float | None = None, ctau: float | None = None) -> float:
        raise NotImplementedError
    
    def get_sigma_left(self, ma: float | None = None, ctau: float | None = None) -> float:
        raise NotImplementedError
    
    def get_sigma_right(self, ma: float | None = None, ctau: float | None = None) -> float:
        raise NotImplementedError
    
    def decay_probability(self, ctau: float | None = None, ma: float | None = None, theta: float | None = None) -> float:
        self.initiate()
        if self.type == 'flat':
            return 1
        pa_parent = np.sqrt(kallen(self.mass_parent**2, ma**2, self.mass_sibling**2))/(2*self.mass_parent)
        if self.lab_boost == 0:
            pa_lab = pa_parent
        else:
            Ea_parent = (self.mass_parent**2 + ma**2 - self.mass_sibling**2)/(2*self.mass_parent)
            lab_gamma = np.sqrt(1 + self.lab_boost**2)
            pa = lambda th: np.sqrt((self.lab_boost * Ea_parent + lab_gamma * pa_parent * np.cos(th))**2 + (pa_parent * np.sin(th))**2)
            if theta is None:
                pa_lab = quad_vec(pa, 0, np.pi)[0]/np.pi
            else:
                pa_lab = pa(theta)
        betagamma = pa_lab/ma
        if self.type == 'prompt':
            return 1 - np.exp(-self.rmin/ctau/betagamma)
        elif self.type == 'displaced':
            return np.exp(-self.rmin/ctau/betagamma) - np.exp(-self.rmax/ctau/betagamma)
        elif self.type == 'invisible':
            return np.exp(-self.rmax/ctau/betagamma)
        
class MeasurementConstant(MeasurementBase):
    def __init__(self, inspire_id: str, type: str, value: float, sigma_left: float, sigma_right: float, min_ma: float=0, rmin: float|None = None, rmax: float|None = None, lab_boost: float = 0.0, mass_parent: float = 0.0, mass_sibling: float = 0.0):
        super().__init__(inspire_id, type, rmin, rmax, lab_boost, mass_parent, mass_sibling)
        self.value = value
        self.sigma_left = sigma_left
        self.sigma_right = sigma_right
        self.min_ma = min_ma
        self.max_ma = self.mass_parent - self.mass_sibling

    def get_central(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), self.value, np.nan)
    
    def get_sigma_left(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), self.sigma_left, np.nan)

    def get_sigma_right(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), self.sigma_right, np.nan)
    
class MeasurementConstantBound(MeasurementConstant):
    def __init__(self, inspire_id: str, type: str, bound: float, min_ma: float = 0, conf_level: float = 0.9, rmin: float | None = None, rmax: float | None = None, lab_boost: float = 0, mass_parent: float = 0, mass_sibling: float = 0):
        super().__init__(inspire_id, type, 0, 0, sigma(conf_level, 1, bound), min_ma, rmin, rmax, lab_boost, mass_parent, mass_sibling)

class MeasurementInterpolatedBound(MeasurementBase):
    def __init__(self, inspire_id, filepath: str, type: str, conf_level: float = 0.9, rmin = None, rmax = None, lab_boost = 0, mass_parent = 0, mass_sibling = 0):
        super().__init__(inspire_id, type, rmin, rmax, lab_boost, mass_parent, mass_sibling)
        self.filepath = filepath
        self.conf_level = conf_level

    def initiate(self):
        super().initiate()
        df = pd.read_csv(self.filepath, sep='\t', header=None)
        self.interpolator = interp1d((df[0]+df[1])/2, df[2], kind='linear')
        self.min_ma = np.min(self.interpolator.x)
        self.max_ma = np.max(self.interpolator.x)

    def get_central(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), 0, np.nan)
    
    def get_sigma_left(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), 0, np.nan)
    
    def get_sigma_right(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        valid_sigmar = self.interpolator(ma[valid_ma])
        sigmar = np.full_like(ma, np.nan)
        sigmar[valid_ma] = valid_sigmar
        return sigmar
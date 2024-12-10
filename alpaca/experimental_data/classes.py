import numpy as np
from scipy.stats import chi2
from ..citations import citations
from ..common import kallen
from ..constants import c_nm_per_ps
from scipy.integrate import quad_vec
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator

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
            if isinstance(self.inspire_id, str):
                citations.register_inspire(self.inspire_id)
            else:
                for inspire_id in self.inspire_id:
                    citations.register_inspire(inspire_id)
        
    def get_central(self, ma: float | None = None, ctau: float | None = None) -> float:
        raise NotImplementedError
    
    def get_sigma_left(self, ma: float | None = None, ctau: float | None = None) -> float:
        raise NotImplementedError
    
    def get_sigma_right(self, ma: float | None = None, ctau: float | None = None) -> float:
        raise NotImplementedError
    
    def decay_probability(self, ctau: float | None = None, ma: float | None = None, theta: float | None = None, br_dark = 0) -> float:
        self.initiate()
        if self.type == 'flat':
            return 1
        kallen_M = kallen(self.mass_parent**2, ma**2, self.mass_sibling**2)
        kallen_M = np.where(kallen_M >0, kallen_M, np.nan)
        pa_parent = np.sqrt(kallen_M)/(2*self.mass_parent)
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
        underflow_error = np.geterr()['under']
        np.seterr(under='ignore')
        if self.type == 'prompt':
            result = 1 - np.exp(-self.rmin/ctau/betagamma)
        elif self.type == 'displaced':
            result = np.exp(-self.rmin/ctau/betagamma) - np.exp(-self.rmax/ctau/betagamma)
        elif self.type == 'invisible':
            br_dark = np.atleast_1d(br_dark)
            prob = np.exp(-self.rmax/ctau/betagamma)
            result = prob + (1 - prob)*br_dark
        np.seterr(under=underflow_error)
        return result
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
        self.interpolator = interp1d((df[0]+df[1])/2, sigma(self.conf_level, 1, df[2]), kind='linear')
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
    
class MeasurementInterpolated(MeasurementBase):
    def __init__(self, inspire_id, filepath: str, type: str, rmin = None, rmax = None, lab_boost = 0, mass_parent = 0, mass_sibling = 0):
        super().__init__(inspire_id, type, rmin, rmax, lab_boost, mass_parent, mass_sibling)
        self.filepath = filepath

    def initiate(self):
        super().initiate()
        df = pd.read_csv(self.filepath, sep='\t', header=None)
        self.interpolator_central = interp1d(df[0], df[1], kind='linear')
        self.interpolator_liminf = interp1d(df[0], df[2], kind='linear')
        self.interpolator_limsup = interp1d(df[0], df[3], kind='linear')
        self.min_ma = np.min(self.interpolator_central.x)
        self.max_ma = np.max(self.interpolator_central.x)

    def get_central(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        valid_central = self.interpolator_central(ma[valid_ma])
        central = np.full_like(ma, np.nan)
        central[valid_ma] = valid_central
        return central
    
    def get_sigma_left(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        valid_liminf = self.interpolator_liminf(ma[valid_ma])
        liminf = np.full_like(ma, np.nan)
        liminf[valid_ma] = valid_liminf
        valid_central = self.interpolator_central(ma[valid_ma])
        central = np.full_like(ma, np.nan)
        central[valid_ma] = valid_central
        return central - liminf
    
    def get_sigma_right(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        valid_limsup = self.interpolator_limsup(ma[valid_ma])
        limsup = np.full_like(ma, np.nan)
        limsup[valid_ma] = valid_limsup
        valid_central = self.interpolator_central(ma[valid_ma])
        central = np.full_like(ma, np.nan)
        central[valid_ma] = valid_central
        return limsup - central
    

class MeasurementDisplacedVertexBound(MeasurementBase):
    def __init__(self, inspire_id, filepath, conf_level: float = 0.9, rmin = None, rmax = None, lab_boost = 0, mass_parent = 0, mass_sibling = 0):
        type = 'displaced'
        super().__init__(inspire_id, type, rmin, rmax, lab_boost, mass_parent, mass_sibling)
        self.filepath = filepath
        self.conf_level = conf_level

    def initiate(self):
        super().initiate()
        data = np.load(self.filepath)
        ma = data[-1,:-1]
        logtau = data[:-1,-1]
        br = data[:-1,:-1]
        self.min_ma = np.min(ma)
        self.max_ma = np.max(ma)
        self.min_tau = 10**np.min(logtau)
        self.max_tau = 10**np.max(logtau)
        self.interpolator = RegularGridInterpolator((ma, logtau), br.T, method='linear', bounds_error=False)

    def get_central(self, ma: float, ctau: float) -> float:
        self.initiate()
        tau = ctau*1e7/c_nm_per_ps
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), 0, np.nan)
    
    def get_sigma_left(self, ma: float, ctau: float) -> float:
        self.initiate()
        tau = ctau*1e7/c_nm_per_ps
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), 0, np.nan)
    
    def get_sigma_right(self, ma: float, ctau: float) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        tau0 = np.atleast_1d(ctau)*1e7/c_nm_per_ps
        shape = np.broadcast_shapes(ma.shape, tau0.shape)
        ma = np.broadcast_to(ma, shape)
        tau0 = np.broadcast_to(tau0, shape)
        tau = np.where(tau0 <= self.max_tau, np.where(tau0 < self.min_tau, self.min_tau, tau0), self.max_tau)
        points = np.vstack((ma.ravel(), np.log10(tau).ravel())).T
        mult = np.where(tau0 <= self.max_tau, 1.0, (1-np.exp(-1))/(1-np.exp(-self.max_tau/tau0)))
        return sigma(self.conf_level, 1, 10**self.interpolator(points).reshape(ma.shape)*mult)
    
    def decay_probability(self, ctau, ma, theta = None, br_dark = 0):
        self.initiate()
        tau = ctau*1e7/c_nm_per_ps
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), 1.0, 0.0)
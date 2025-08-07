import numpy as np
from scipy.stats import chi2
from ..biblio.biblio import citations
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

def sigma(cl, param):
    return np.sqrt(2/np.pi)*param*(1-cl)/cl

class MeasurementBase:
    """
    Base class for measurements.
    This class provides a common interface for different types of measurements.
    It is not intended to be instantiated directly.

    Attributes:

        inspire_id (str): Inspire-HEP reference of the measurement.
        decay_type (str): The decay_type of the instance.
        rmin (float|None): The minimum length of the detector.
        rmax (float|None): The maximum length of the detector.
        initiated (bool): Indicates if the instance has been initiated.
        lab_boost (float): The laboratory boost value.
        mass_parent (float): The mass of the parent.
        mass_sibling (float): The mass of the sibling.
        conf_level (float|None): The confidence level of the measurement.

    Methods:

        initiate(): Initializes the instance by registering the Inspire-HEP reference.
        get_central(ma: float|None, ctau: float|None): Returns the central value of the measurement.
        get_sigma_left(ma: float|None, ctau: float|None): Returns the left sigma of the measurement.
        get_sigma_right(ma: float|None, ctau: float|None): Returns the right sigma of the measurement.
        decay_probability(ctau: float|None, ma: float|None, br_dark: float=0.0): Calculates the decay probability for a given mass and ctau.
    """
    def __init__(self, inspire_id: str, decay_type: str, rmin: float|None = None, rmax: float|None = None, lab_boost: float = 0.0, mass_parent: float = 0.0, mass_sibling: float = 0.0, bibtex: dict[str,str] | None = None):
        """
        Initialize an instance of the class.

        Parameters:
        inspire_id (str): Inspire-HEP reference of the measurement.
        decay_type (str): The decay_type of the instance.
        rmin (float, optional): The minimum radius. Defaults to None.
        rmax (float, optional): The maximum radius. Defaults to None.
        lab_boost (float): The laboratory boost value. Defaults to 0.0.
        mass_parent (float): The mass of the parent. Defaults to 0.0.
        mass_sibling (float): The mass of the sibling. Defaults to 0.0.
        """
        self.inspire_id = inspire_id
        self.decay_type = decay_type
        self.rmin = rmin
        self.rmax = rmax
        self.initiated = False
        self.lab_boost = lab_boost
        self.mass_parent = mass_parent
        self.mass_sibling = mass_sibling
        self.conf_level = None
        self.bibtex = bibtex

    def initiate(self):
        if not self.initiated:
            self.initiated = True
            if isinstance(self.inspire_id, str):
                citations.register_inspire(self.inspire_id)
            else:
                for inspire_id in self.inspire_id:
                    citations.register_inspire(inspire_id)
            if self.bibtex is not None:
                for key, value in self.bibtex.items():
                    citations.register_bibtex(key, value)

    def get_central(self, ma: float | None = None, ctau: float | None = None) -> float:
        '''
        Get the central value of the measurement.

        Parameters
        ----------
        ma : float | None
            The mass of the alp, in GeV.
        ctau : float | None
            The proper length of the alp, in cm.
        '''
        raise NotImplementedError
    
    def get_sigma_left(self, ma: float | None = None, ctau: float | None = None) -> float:
        '''
        Get the left one-sided uncertainty of the measurement.
        In the case of an upper limit bound, this is zero.

        Parameters
        ----------
        ma : float | None
            The mass of the alp, in GeV.
        ctau : float | None
            The proper length of the alp, in cm.
        '''
        raise NotImplementedError
    
    def get_sigma_right(self, ma: float | None = None, ctau: float | None = None) -> float:
        '''
        Get the right one-sided uncertainty of the measurement.

        Parameters
        ----------
        ma : float | None
            The mass of the alp, in GeV.
        ctau : float | None
            The proper length of the alp, in cm.
        '''
        raise NotImplementedError
    
    def decay_probability(self, ctau: float | None = None, ma: float | None = None, br_dark: float = 0.0, theta: float | None = None) -> float:
        '''
        Calculate the probability for the decay of the alp corresponding to the decay_type.

        Parameters
        ----------
        ctau : float | None
            The proper length of the alp, in cm.
        ma : float | None
            The mass of the alp, in GeV.
        br_dark : float
            The branching ratio of decay into the dark sector.
        '''
        self.initiate()
        if self.decay_type == 'flat':
            return 1
        ma = np.atleast_1d(ma)
        ctau = np.atleast_1d(ctau)
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
        if self.decay_type == 'prompt':
            result = 1 - np.exp(-self.rmin/ctau/betagamma)
        elif self.decay_type == 'displaced':
            result = np.exp(-self.rmin/ctau/betagamma) - np.exp(-self.rmax/ctau/betagamma)
        elif self.decay_type == 'invisible':
            br_dark = np.atleast_1d(br_dark)
            prob = np.exp(-self.rmax/ctau/betagamma)
            result = prob + (1 - prob)*br_dark
        np.seterr(under=underflow_error)
        return result
class MeasurementConstant(MeasurementBase):
    def __init__(self, inspire_id: str, decay_type: str, value: float, sigma_left: float, sigma_right: float, min_ma: float=0, max_ma : float|None = None, rmin: float|None = None, rmax: float|None = None, lab_boost: float = 0.0, mass_parent: float = 0.0, mass_sibling: float = 0.0, bibtex: dict[str,str] | None = None):
        super().__init__(inspire_id, decay_type, rmin, rmax, lab_boost, mass_parent, mass_sibling, bibtex)
        self.value = value
        self.sigma_left = sigma_left
        self.sigma_right = sigma_right
        self.min_ma = min_ma
        if max_ma is None:
            self.max_ma = self.mass_parent - self.mass_sibling
        else:
            self.max_ma = max_ma

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
    def __init__(self, inspire_id: str, decay_type: str, bound: float, min_ma: float = 0, max_ma: float|None = None, conf_level: float = 0.9, rmin: float | None = None, rmax: float | None = None, lab_boost: float = 0, mass_parent: float = 0, mass_sibling: float = 0, bibtex: dict[str,str] | None = None):
        super().__init__(inspire_id, decay_type, bound, 0, sigma(conf_level, bound), min_ma, max_ma, rmin, rmax, lab_boost, mass_parent, mass_sibling, bibtex)
        self.conf_level = conf_level

class MeasurementInterpolatedBound(MeasurementBase):
    def __init__(self, inspire_id, filepath: str, decay_type: str, conf_level: float = 0.9, rmin = None, rmax = None, lab_boost = 0, mass_parent = 0, mass_sibling = 0, bibtex: dict[str,str] | None = None):
        super().__init__(inspire_id, decay_type, rmin, rmax, lab_boost, mass_parent, mass_sibling, bibtex)
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
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        valid_value = self.interpolator(ma[valid_ma])
        value = np.full_like(ma, np.nan)
        value[valid_ma] = valid_value
        return value
    
    def get_sigma_left(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), 0, np.nan)
    
    def get_sigma_right(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        return sigma(self.conf_level, self.get_central(ma, ctau))
    
class MeasurementInterpolated(MeasurementBase):
    def __init__(self, inspire_id, filepath: str, decay_type: str, rmin = None, rmax = None, lab_boost = 0, mass_parent = 0, mass_sibling = 0, bibtex: dict[str,str] | None = None):
        super().__init__(inspire_id, decay_type, rmin, rmax, lab_boost, mass_parent, mass_sibling, bibtex)
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
    
class MeasurementBinned(MeasurementBase):
    def __init__(self, inspire_id, filepath, decay_type, rmin = None, rmax = None, lab_boost = 0, mass_parent = 0, mass_sibling = 0, bibtex: dict[str,str] | None = None):
        super().__init__(inspire_id, decay_type, rmin, rmax, lab_boost, mass_parent, mass_sibling, bibtex)
        self.filepath = filepath

    def initiate(self):
        if self.initiated:
            return
        super().initiate()
        meas = {}
        with open(self.filepath, 'rt') as f:
            l = f.readlines()
        for line in l:
            vals = [float(v) for v in line.split()]
            meas |= {(vals[0], vals[1]): (vals[2], vals[3], vals[4])}
        gaps = {}
        for i, k in enumerate(meas.keys()):
            if i == 0:
                prev_k = k
                continue
            if prev_k[1] != k[0]:
                gaps |= {(prev_k[1], k[0]): (np.nan, np.nan, np.nan)}
            prev_k = k
        meas |= gaps
        bins = sorted(meas, key=lambda x: x[0])
        data = {binn: meas[binn] for binn in bins}
        self.bin_limits = [b[0] for b in data.keys()] + [list(data.keys())[-1][1]]
        self.centrals = [v[0] for v in data.values()]
        self.sigma_l = [v[1] for v in data.values()]
        self.sigma_r = [v[2] for v in data.values()]
        self.min_ma = np.min(self.bin_limits)
        self.max_ma = np.max(self.bin_limits)

    def get_central(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        indexes = np.digitize(ma[valid_ma], self.bin_limits)
        central = np.full(ma.shape, np.nan)
        central[valid_ma] = np.array([self.centrals[ix] for ix in indexes])
        return central

    def get_sigma_left(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        indexes = np.digitize(ma[valid_ma], self.bin_limits)
        sigmas = np.full(ma.shape, np.nan)
        sigmas[valid_ma] = np.array([self.sigma_l[ix] for ix in indexes])
        return sigmas
    
    def get_sigma_right(self, ma: float, ctau: float | None = None) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        valid_ma = np.where((ma >= self.min_ma) & (ma <= self.max_ma))
        indexes = np.digitize(ma[valid_ma], self.bin_limits)
        sigmas = np.full(ma.shape, np.nan)
        sigmas[valid_ma] = np.array([self.sigma_r[ix] for ix in indexes])
        return sigmas
class MeasurementDisplacedVertexBound(MeasurementBase):
    def __init__(self, inspire_id, filepath, conf_level: float = 0.9, rmin = None, rmax = None, lab_boost = 0, mass_parent = 0, mass_sibling = 0, decay_type = 'displaced', bibtex: dict[str,str] | None = None):
        decay_type = decay_type
        super().__init__(inspire_id, decay_type, rmin, rmax, lab_boost, mass_parent, mass_sibling, bibtex)
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
        self.interpolator = RegularGridInterpolator((ma, logtau), np.nan_to_num(br.T, nan=1000), method='linear', bounds_error=False)

    def get_central(self, ma: float, ctau: float) -> float:
        self.initiate()
        ma = np.atleast_1d(ma)
        tau0 = np.atleast_1d(ctau)*1e7/c_nm_per_ps
        shape = np.broadcast_shapes(ma.shape, tau0.shape)
        ma = np.broadcast_to(ma, shape)
        tau0 = np.broadcast_to(tau0, shape)
        tau = np.where(tau0 <= self.max_tau, np.where(tau0 < self.min_tau, self.min_tau, tau0), self.max_tau)
        points = np.vstack((ma.ravel(), np.log10(tau).ravel())).T
        logbr = self.interpolator(points).reshape(shape)
        logbr = np.where(logbr < 0, logbr, np.nan)
        return 10**logbr
    
    def get_sigma_left(self, ma: float, ctau: float) -> float:
        self.initiate()
        tau = ctau*1e7/c_nm_per_ps
        return np.where((ma >= self.min_ma) & (ma <= self.max_ma), 0, np.nan)
    
    def get_sigma_right(self, ma: float, ctau: float) -> float:
        self.initiate()
        return sigma(self.conf_level, self.get_central(ma, ctau))
    
    def get_values(self, ma: float, ctau: float) -> float:
        self.initiate()
        return self.get_central(ma, ctau), self.get_sigma_left(ma, ctau), self.get_sigma_right(ma, ctau)
    
    def decay_probability(self, ctau, ma, theta = None, br_dark = 0):
        self.initiate()
        tau = np.atleast_1d(ctau)*1e7/c_nm_per_ps
        ma = np.atleast_1d(ma)
        shape = np.broadcast_shapes(ma.shape, tau.shape)
        ma = np.broadcast_to(ma, shape)
        tau = np.broadcast_to(tau, shape)
        if self.decay_type == 'displaced':
            prob = np.where(tau <= self.max_tau, 1.0, (1-np.exp(-self.max_tau/tau))/(1-np.exp(-1)))
            return np.where((ma >= self.min_ma) & (ma <= self.max_ma), prob, 0.0)
        elif self.decay_type == 'invisible':
            br_dark = np.atleast_1d(br_dark)
            prob = np.where(tau >= self.min_tau, 1.0, np.exp(-self.min_tau/tau)/np.exp(-1))
            return np.where((ma >= self.min_ma) & (ma <= self.max_ma), prob + (1 - prob)*br_dark, 0.0)
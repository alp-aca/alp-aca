import numpy as np 
import scipy.stats

def tensor_meshgrid(*arrays):
    dims = [np.array(a).shape for a in arrays]
    if dims == [(),]*len(arrays):
        return arrays
    dims_final = [x for xs in dims for x in xs]
    ones = [np.ones_like(d).tolist() for d in dims]
    result = []
    for i, a in enumerate(arrays):
        dd = [d for d in ones]
        dd[i] = dims[i]
        dim_reshape = [x for xs in dd for x in xs]
        if dim_reshape == []:
            dim_reshape = [1]
        result.append(np.broadcast_to(np.array(a).reshape(*dim_reshape), dims_final))
    return tuple(result)

def nsigmas(chi2: np.ndarray[float], ndof: np.ndarray[float]) -> np.ndarray[float]:
    r"""Compute the pull in Gaussian standard deviations corresponding to
    a $\chi^2$ with `ndof` degrees of freedom.

    Example: For `dof=2` and `delta_chi2=2.3`, the result is roughly 1.0."""
    p = 1 - scipy.stats.chi2.cdf(np.where(ndof == 0, np.nan, chi2), ndof)
    p = np.clip(p, 2e-16, 1)
    return scipy.stats.norm.ppf(1 - p/2)
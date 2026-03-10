import numpy as np 
import scipy.stats
from scipy.special import gammaln

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

@np.vectorize
def nsigmas(chi2: float, ndof: float) -> float:
    r"""Compute the pull in Gaussian standard deviations corresponding to
    a $\chi^2$ with `ndof` degrees of freedom.
    """
    if ndof == 0:
        return np.nan
    if ndof == 1:
        return np.sqrt(chi2)
    logsf = scipy.stats.chi2.logsf(chi2, ndof) - np.log(2)
    if np.isinf(logsf):
        logsf = -chi2/2 + (ndof-1) * np.log(chi2) - ndof/2 * np.log(2) - gammaln(ndof/2) - np.log(2)
    if logsf > -720:
        return scipy.stats.norm.isf(np.exp(logsf))
    else:
        z = np.sqrt(-2* logsf)
        for _ in range(5):
            z = np.sqrt(-2*(logsf + 0.5*np.log(2*np.pi) + np.log(z)))
        return z
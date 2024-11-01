import numpy as np 
import scipy.stats

def tensor_meshgrid(*arrays):
    dims = [np.array(a).shape for a in arrays]
    dims_final = [x for xs in dims for x in xs]
    ones = [np.ones_like(d).tolist() for d in dims]
    result = []
    for i, a in enumerate(arrays):
        dd = [d for d in ones]
        dd[i] = dims[i]
        dim_reshape = [x for xs in dd for x in xs]
        result.append(np.broadcast_to(np.array(a).reshape(*dim_reshape), dims_final))
    return result

def nsigmas(chi2, ndof):
    r"""Compute the pull in Gaussian standard deviations corresponding to
    a $\chi^2$ with `ndof` degrees of freedom.

    Example: For `dof=2` and `delta_chi2=2.3`, the result is roughly 1.0."""
    if ndof == 1:
        # that's trivial
        return np.sqrt(abs(chi2))
    chi2_ndof = scipy.stats.chi2(ndof)
    cl_delta_chi2 = chi2_ndof.cdf(chi2)
    sigmas = scipy.stats.norm.ppf(0.5+cl_delta_chi2/2)
    return np.where(np.isinf(sigmas), 1e6, sigmas)
import numpy as np
import sympy as sp

def almost_real(x: complex) -> bool:
    """Check if a complex number is almost real."""
    if np.imag(x) == 0:
        return True
    return bool(np.isclose(np.angle(x), 0, atol=1e-3) or np.isclose(np.abs(np.angle(x)), np.pi, atol=1e-3))


def is_zero_matrix_sp(m: sp.Matrix, r:int = 3) -> bool:
    """Check if a sympy matrix is a zero matrix."""
    return all((m[i, j] == 0 or m[i, j] == 0.0) for i in range(r) for j in range(r))
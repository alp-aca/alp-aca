import numpy as np
import gvar
from ..rge import ALPcouplings
import os

def det3(M):
    return M[0,0] * (M[1,1] * M[2,2] - M[1,2] * M[2,1]) - M[0,1] * (M[1,0] * M[2,2] - M[1,2]*M[2,0]) + M[0,2]*(M[1,0]*M[2,1] - M[1,1]*M[2,0])


def eigh_3x3_analytic_np(M):
    """
    Analytically diagonalize a real symmetric 3x3 matrix.
    Returns eigenvalues (ascending) and eigenvectors as rows of R.
    
    Uses the method of: Smith (1961), as used in e.g. Kopp (2008)
    arxiv:physics/0610206 — numerically stable, smooth gradients.
    """
    p1 = M[0,1]**2 + M[0,2]**2 + M[1,2]**2

    # If p1 ~ 0 the matrix is already diagonal
    # We don't branch (not allowed in pytensor) but the formula
    # reduces correctly in that limit

    q  = np.trace(M) / 3.0
    Mq = M - q * np.eye(3)
    
    p2 = (Mq[0,0]**2 + Mq[1,1]**2 + Mq[2,2]**2
          + 2*p1)
    p  = np.sqrt(p2 / 6.0)

    B  = Mq / p                    # B = (M - q*I) / p
    r  = det3(B) / 2.0

    # r is in [-1, 1] analytically; clamp for numerical safety
    # Use a smooth clamp: arcsin is undefined outside [-1,1]
    r_clamped = np.clip(r, -1.0 + 1e-6, 1.0 - 1e-6)
    phi = np.arccos(r_clamped) / 3.0

    # Eigenvalues in ascending order
    eig0 = q + 2 * p * np.cos(phi + 2*np.pi/3)
    eig1 = q + 2 * p * np.cos(phi + 4*np.pi/3)
    eig2 = q + 2 * p * np.cos(phi)

    # Eigenvectors: for each eigenvalue, compute (M - lambda*I) and
    # take the cross product of any two rows (most stable pair)
    def eigvec(lam):
        A = M - lam * np.eye(3)
        # Cross products of row pairs
        v0 = np.cross(A[0], A[1])
        v1 = np.cross(A[0], A[2])
        v2 = np.cross(A[1], A[2])
        # Pick the longest (most numerically stable)
        n0 = gvar.mean(np.dot(v0, v0))
        n1 = gvar.mean(np.dot(v1, v1))
        n2 = gvar.mean(np.dot(v2, v2))
        if n0 >= max(n1, n2):
            v = v0
        elif n1 >= n2:
            v = v1
        else:
            v = v2
        return v / np.sqrt(np.dot(v, v) + 1e-30)

    r0 = eigvec(eig0)
    r1 = eigvec(eig1)
    r2 = eigvec(eig2)

    eigenvalues  = np.stack([eig0, eig1, eig2])
    eigenvectors = np.stack([r0, r1, r2])     # rows are eigenvectors

    return eigenvalues, eigenvectors

def mass_sol(B0mu, B0md, B0ms, m02, **kwargs):
    M_bare = np.array([
        [B0mu + B0md, (B0mu - B0md) / np.sqrt(3.0), (B0mu - B0md) / np.sqrt(1.5)],
        [(B0mu - B0md) / np.sqrt(3.0), (B0mu + B0md + 4*B0ms) / 3.0, np.sqrt(2.0) / 3.0 * (B0mu + B0md - 2*B0ms)],
        [(B0mu - B0md) / np.sqrt(1.5), np.sqrt(2.0) / 3.0 * (B0mu + B0md - 2*B0ms), 2*(B0mu + B0md + B0ms) / 3.0 + m02],
    ])

    # Exact diagonalization — no angular parameters, no Potential
    return eigh_3x3_analytic_np(M_bare)

class FitResults:
    def __init__(self):
        self.initialized = False
    
    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        current_path = os.path.dirname(__file__)
        fit_res = gvar.load(os.path.join(current_path, 'mesonfit.pickle'))

        mneutr, R = mass_sol(**fit_res)
        lam_pi3 = np.diag([1, -1, 0])
        lam_eta8 = np.diag([1, 1, -2])/np.sqrt(3)
        lam_eta0 = np.diag([1, 1, 1])/np.sqrt(3/2)

        self.lam_pi0 = gvar.mean(R[0,0] * lam_pi3 + R[0,1] * lam_eta8 + R[0,2] * lam_eta0)
        self.lam_eta = gvar.mean(R[1,0] * lam_pi3 + R[1,1] * lam_eta8 + R[1,2] * lam_eta0)
        self.lam_etap = gvar.mean(R[2,0] * lam_pi3 + R[2,1] * lam_eta8 + R[2,2] * lam_eta0)
        self.B0mu = gvar.mean(fit_res['B0mu'])
        self.B0md = gvar.mean(fit_res['B0md'])
        self.B0ms = gvar.mean(fit_res['B0ms'])
        self.m02 = gvar.mean(fit_res['m02'])

    def __getitem__(self, key):
        self.initialize()
        if key == 'lam_pi0':
            return self.lam_pi0
        elif key == 'lam_eta':
            return self.lam_eta
        elif key == 'lam_etap':
            return self.lam_etap
        elif key == 'B0mu':
            return self.B0mu
        elif key == 'B0md':
            return self.B0md
        elif key == 'B0ms':
            return self.B0ms
        elif key == 'm02':
            return self.m02
        elif key == 'B0mq':
            return np.diag([self.B0mu, self.B0md, self.B0ms])
        else:
            raise KeyError(f"Key {key} not found")

fit_results = FitResults()

def coupling_mixing(couplings: ALPcouplings):
    couplings = couplings.translate('VA_below')
    cuA = couplings['cuA']
    cdA = couplings['cdA']
    cqA = np.array([[cuA[0,0], 0, 0], [0, cdA[0,0], cdA[0,1]], [0, cdA[1,0], cdA[1,1]]])/2
    return fit_results['B0mq'] @ cqA + cqA @ fit_results['B0mq'] - fit_results['m02']/3 * np.eye(3) * couplings['cG']

def mixing_shift(couplings: ALPcouplings, ma: float):
    # This is \mathbb{M}_a^{-1}(C)
    C = coupling_mixing(couplings)
    B0mq = fit_results['B0mq']
    m02 = fit_results['m02']
    poles = np.zeros((3,3), dtype=object)
    for i in range(3):
        for j in range(3):
            poles[i,j] = 1/(B0mq[i,i]+B0mq[j,j]-ma**2)
    poles_diag = np.diag(np.diag(poles))
    return C * poles - poles_diag * np.trace(C @ poles_diag)/(3/m02 + np.trace(poles_diag)) # C * poles is a Hadamard (element-wise) product!
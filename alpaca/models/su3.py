import sympy as sp
from ..citations import citations

def dim_from_dynkinlabels(l1: int, l2: int) -> int:
    """Returns the dimension of the SU(3) representation with the given Dynkin labels."""
    citations.register_bibtex('Slansky', reference_Slansky)
    return int((l1+1)*(l2+1)*(l1+l2+2)/2)

def dynkinlabels_from_dim(dim: int) -> set[tuple[int, int]]:
    """Returns the Dynkin labels of the SU(3) representation with the given dimension."""
    maxlabel = int((-3+(1+8*dim)**0.5)/2)
    labels = set()
    for l1 in range(maxlabel+1):
        for l2 in range(l1+1):
            if dim_from_dynkinlabels(l1, l2) == dim:
                labels.add((l1, l2))
                labels.add((l2, l1))
    return labels

def index_from_dynkinlabels(l1: int, l2: int) -> float:
    """Returns the Dynkin index of the SU(3) representation with the given Dynkin labels."""
    citations.register_bibtex('Slansky', reference_Slansky)
    return sp.Rational(1,2)*int(dim_from_dynkinlabels(l1, l2)*(l1**2+3*l1+l1*l2+3*l2+l2**2)/12)

def dynkinlabels_from_name(name: str) -> tuple[int, int]:
    """Returns the Dynkin labels of the SU(3) representation with the given name."""
    if isinstance(name, int):
        dim = name
        primes = 0
        if dim < 0:
            bar = True
            dim = -dim
        else:
            bar = False
    else:
        if name.endswith('_bar'):
            bar = True
            name = name[:-4]
        else:
            bar = False
        primes = sum(1 for c in name if c == "'")
        dim = int(name.replace("'", ""))
    reprs = dynkinlabels_from_dim(dim)
    if not reprs:
        raise KeyError(f"The representation {name} of the group SU(3) does not exist.")
    if bar:
        reprs = [repr for repr in reprs if repr[1] >= repr[0]]
        reprs = sorted(reprs, key=lambda x: x[1])
    else:
        reprs = [repr for repr in reprs if repr[0] >= repr[1]]
        reprs = sorted(reprs, key=lambda x: x[0])
    if primes > len(reprs)-1:
        raise KeyError(f"The representation {name} of the group SU(3) does not exist.")
    return reprs[primes]

reference_Slansky = '''@article{Slansky:1981yr,
    author = "Slansky, R.",
    title = "{Group Theory for Unified Model Building}",
    reportNumber = "LA-UR-80-3495",
    doi = "10.1016/0370-1573(81)90092-2",
    journal = "Phys. Rept.",
    volume = "79",
    pages = "1--128",
    year = "1981"
}
'''
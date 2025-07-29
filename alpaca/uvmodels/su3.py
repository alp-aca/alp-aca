import sympy as sp
from ..biblio.biblio import citations

def dim_from_dynkinlabels(l1: int, l2: int) -> int:
    """Returns the dimension of the SU(3) representation with the given Dynkin labels."""
    citations.register_inspire('Slansky:1981yr')
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
    citations.register_inspire('Slansky:1981yr')
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

def name_from_dynkinlabels(l1: int, l2: int) -> str:
    if l2 > l1:
        return name_from_dynkinlabels(l2, l1) + "_bar"
    dim = dim_from_dynkinlabels(l1, l2)
    reprs = dynkinlabels_from_dim(dim)
    reprs = [repr for repr in reprs if repr[0] >= repr[1]]
    reprs = sorted(reprs, key=lambda x: x[0])
    index = reprs.index((l1, l2))
    return str(dim) + "'" * index

def latex_from_dynkinlabels(l1: int, l2: int) -> str:
    """Returns the LaTeX representation of the SU(3) representation with the given Dynkin labels."""
    name = name_from_dynkinlabels(l1, l2)
    if name.endswith('_bar'):
        name = name[:-4]
        bar = True
    else:
        bar = False
    primes = name.count("'")
    name = name.replace("'", "")
    if bar:
        num = f"$\\overline{{\\mathbf{{{name}}}}}"
    else:
        num = f"$\\mathbf{{{name}}}"
    if primes > 0:
        num += f"^{{{'\\prime' * primes}}}$"
    else:
        num += "$"
    return num
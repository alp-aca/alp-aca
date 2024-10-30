class LazyFloat:
    def __init__(self, func):
        self._func = func
        self._value = None

    def _evaluate(self):
        if self._value is None:
            self._value = self._func()
        return self._value

    def __repr__(self):
        return f"{self.__class__.__name__}({self._evaluate()})"

    def __str__(self):
        return str(self._evaluate())

    def __complex__(self):
        return complex(self._evaluate())

    def __add__(self, other):
        return complex(self) + complex(other)

    def __radd__(self, other):
        return complex(other) + complex(self)

    def __sub__(self, other):
        return complex(self) - complex(other)

    def __rsub__(self, other):
        return complex(other) - complex(self)

    def __mul__(self, other):
        return complex(self) * complex(other)

    def __rmul__(self, other):
        return complex(other) * complex(self)

    def __truediv__(self, other):
        return complex(self) / complex(other)

    def __rtruediv__(self, other):
        return complex(other) / complex(self)
    
    def __pow__(self, other):
        return complex(self)**other

    # Comparison operators
    def __eq__(self, other):
        return complex(self) == complex(other)

    def __lt__(self, other):
        return complex(self) < complex(other)

    def __le__(self, other):
        return complex(self) <= complex(other)

    def __gt__(self, other):
        return complex(self) > complex(other)

    def __ge__(self, other):
        return complex(self) >= complex(other)
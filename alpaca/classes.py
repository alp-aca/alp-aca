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

    def __float__(self):
        return float(self._evaluate())

    def __add__(self, other):
        return float(self) + other

    def __radd__(self, other):
        return other + float(self)

    def __sub__(self, other):
        return float(self) - other

    def __rsub__(self, other):
        return other - float(self)

    def __mul__(self, other):
        return float(self) * other

    def __rmul__(self, other):
        return other * float(self)
    
    def __neg__(self):
        return -float(self)

    def __truediv__(self, other):
        return float(self) / other

    def __rtruediv__(self, other):
        return other / float(self)
    
    def __pow__(self, other):
        return float(self)**other

    # Comparison operators
    def __eq__(self, other):
        return float(self) == other

    def __lt__(self, other):
        return float(self) < other

    def __le__(self, other):
        return float(self) <= other

    def __gt__(self, other):
        return float(self) > other

    def __ge__(self, other):
        return float(self) >= other
import requests
import tempfile
from contextlib import contextmanager
from os import PathLike
from io import TextIOBase

class Citations:
    def __init__(self):
        self.citations = set()
        self.dict_citations = {}

    def register_inspire(self, inspires_id: str):
        self.citations |= {inspires_id}

    def register_bibtex(self, key:str, val:str):
        self.dict_citations |= {key: val}

    def register_particle(self):
        bibtex = r"""@software{Rodrigues_Particle,
    author = {Rodrigues, Eduardo and Schreiner, Henry},
    doi = {10.5281/zenodo.2552429},
    license = {BSD-3-Clause},
    title = {{Particle}},
    url = {https://github.com/scikit-hep/particle}
}"""
        self.register_bibtex('particle', bibtex)
        self.register_inspire('ParticleDataGroup:2024cfk')

    def inspires_ids(self):
        return list(self.citations)
    
    def generate_bibtex(self, filepath: str | PathLike | TextIOBase):
        '''Generates a bibtex file with the citations registered in the object.

        This method uses the inspirehep.net API to generate the bibtex file.

        Arguments
        ---------
        filepath: str | PathLike | TextIOBase
            The path to the file where the bibtex will be written. If a file-like
            object is passed, the bibtex will be written to it. If a string is
            passed, the bibtex will be written to a file with the given name.
        '''
        with tempfile.NamedTemporaryFile('w+t', suffix='.tex') as tf:
            tf.write(r'\cite{' + ','.join(citations.inspires_ids()) + r'}')
            tf.seek(0)
            print(tf.read())
            tf.seek(0)
            r = requests.post('https://inspirehep.net/api/bibliography-generator?format=bibtex', files={'file': tf})
        r.raise_for_status()
        r2 = requests.get(r.json()['data']['download_url'], stream=True)
        r2.raise_for_status()
        if isinstance(filepath, TextIOBase):
            for chunck in r2.iter_content(chunk_size=16*1024):
                filepath.write(chunck)
            for v in self.dict_citations.values():
                filepath.write('\n\n' + v)
        else:
            with open(filepath, 'wb') as f:
                for chunck in r2.iter_content(chunk_size=16*1024):
                    f.write(chunck)
                for v in self.dict_citations.values():
                    f.write(str.encode('\n\n' + v))

citations = Citations()
citations.register_inspire('Harris:2020xlr') # Including numpy by default

@contextmanager
def citations_context(merge: bool = True):
    saved_citations = citations.citations.copy()
    saved_dict_citations = citations.dict_citations.copy()
    citations.citations.clear()
    citations.dict_citations.clear()
    yield
    if merge:
        citations.citations |= saved_citations
        citations.dict_citations |= saved_dict_citations
    else:
        citations.citations = saved_citations
        citations.dict_citations = saved_dict_citations

citations_context.__doc__ =    '''Creates a context manager to gather citations in a block of code.
    
    Arguments
    ---------
    merge: bool (default: True)
        If True, the citations gathered in the block will be merged with the
        citations outside the block. If False, the citations gathered in the
        block wil be erased after the block.

    Usage
    -----
    >>> from alpaca.citations import citations, citations_context
    >>> with citations_context():
    >>>     # Code that uses alpaca
    >>>     citations.generate_bibtex('my_bibtex.bib')
    '''

class Constant(float):
    def __new__(self, val: float, source: str):
        return float.__new__(self, val)
    def __init__(self, val: float, source: str):
        self.__dict__['_value'] = val
        self.__dict__['source'] = source
    def register(self):
        if self.source == 'flavio':
            citations.register_inspire('Straub:2018kue')
        elif self.source == 'particle':
            citations.register_particle()
        else:
            citations.register_inspire(self.source)

    def __getnewargs__(self) -> tuple:
        return (self.__dict__['_value'], self.__dict__['source'])
    
    def __reduce__(self) -> tuple:
        return (self.__class__, self.__getnewargs__(), self.__dict__)
    
    def __getattribute__(self, name):
        if name == '_value':
            self.register()
        return super().__getattribute__(name)
    
    def __setattr__(self, name, value):
        raise AttributeError("Constant objects are read-only")
    
    def __add__(self, other):
        self.register()
        return super().__add__(other)
    
    def __radd__(self, other):
        self.register()
        return super().__radd__(other)
    
    def __sub__(self, other):
        self.register()
        return super().__sub__(other)
    
    def __rsub__(self, other):
        self.register()
        return super().__rsub__(other)
    
    def __mul__(self, other):
        self.register()
        return super().__mul__(other)
    
    def __rmul__(self, other):
        self.register()
        return super().__rmul__(other)
    
    def __truediv__(self, other):
        self.register()
        return super().__truediv__(other)
    
    def __rtruediv__(self, other):
        self.register()
        return super().__rtruediv__(other)
    
    def __str__(self):
        self.register()
        return str(self._value)
    
    def __repr__(self):
        self.register()
        return f"Constant({self._value},{self.source})"
    
    def __complex__(self):
        self.register()
        return self._value
        
    # Implement comparison methods
    def __eq__(self, other):
        self.register()
        return self._value == other
    
    def __lt__(self, other):
        self.register()
        return self._value < other
    
    def __le__(self, other):
        self.register()
        return self._value <= other
    
    def __gt__(self, other):
        self.register()
        return self._value > other
    
    def __ge__(self, other):
        self.register()
        return self._value >= other
    
    def __hash__(self) -> int:
        return super().__hash__()

    def __complex__(self) -> complex:
        self.register()
        return float(self) + 0j
import requests
import tempfile

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
    
    def generate_bibtex(self, filepath: str):
        with tempfile.NamedTemporaryFile('w+t', suffix='.tex') as tf:
            tf.write(r'\cite{' + ','.join(citations.inspires_ids()) + r'}')
            tf.seek(0)
            print(tf.read())
            tf.seek(0)
            r = requests.post('https://inspirehep.net/api/bibliography-generator?format=bibtex', files={'file': tf})
        r.raise_for_status()
        r2 = requests.get(r.json()['data']['download_url'], stream=True)
        r2.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunck in r2.iter_content(chunk_size=16*1024):
                f.write(chunck)
            for v in self.dict_citations.values():
                f.write(str.encode('\n\n' + v))

citations = Citations()
citations.register_inspire('Harris:2020xlr') # Including numpy by default

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

import numpy as np

class DMCandidate:
    def __init__(self,
                 name: str,
                 tex_name: str,
                 mathematica_name: str,
                 mass: float,
                 coupling: float):
        self.name = name
        self.tex_name = tex_name
        self.mathematica_name = mathematica_name
        self.mass = mass
        self.coupling = coupling

    def decay_width(self, ma, fa):
        pass

class DMMajorana(DMCandidate):
    def __init__(self,
                 mass: float,
                 coupling: float):
        super().__init__(name="Majorana fermion",
                         tex_name=r"\chi",
                         mathematica_name=r"[\Chi]",
                         mass=mass,
                         coupling=coupling)

    def decay_width(self, ma, fa):
        # Decay width for Majorana DM candidate
        return np.abs(self.coupling)**2 * self.mass**2 * ma / (8 * np.pi * fa**2) * np.sqrt(1 - (2*ma/self.mass)**2)
    
    def __repr__(self) -> str:
        return f"DMMajorana(mass={self.mass}, coupling={self.coupling})"
    
    def _repr_markdown_(self):
        return f"**Dark Matter Majorana fermion**:\n\tmass = {self.mass}\n\tcoupling = {self.coupling}"
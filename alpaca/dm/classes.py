import numpy as np

class DMCandidate:
    def __init__(self,
                 name: str,
                 tex_name: str,
                 tex_decay: str,
                 mathematica_name: str,
                 mass: float,
                 coupling: float):
        self.name = name
        self.tex_name = tex_name
        self.tex_decay = tex_decay
        self.mathematica_name = mathematica_name
        self.mass = mass
        self.coupling = coupling

    def decay_width(self, ma, fa):
        pass

class DMBranchingRatio(DMCandidate):
    def __init__(self,
                 br: float):
        if br < 0.0 or br > 1.0:
            raise ValueError("Branching ratio must be between 0 and 1.")
        super().__init__(name="Generic DM candidate (fixed BR)",
                         tex_name=r'\mathrm{DM}',
                         tex_decay = r'\mathrm{DM}',
                         mathematica_name=r"DM",
                         mass=0.0,
                         coupling=br)

class DMDecayWidth(DMCandidate):
    def __init__(self,
                 dw: float):
        if dw < 0.0:
            raise ValueError("Decay width must be non-negative.")
        super().__init__(name="Generic DM candidate (fixed decay width)",
                         tex_name=r'\mathrm{DM}',
                         tex_decay = r'\mathrm{DM}',
                         mathematica_name=r"DM",
                         mass=0.0,
                         coupling=dw)
        
    def decay_width(self, ma, fa):
        return self.coupling
class DMMajorana(DMCandidate):
    def __init__(self,
                 mass: float,
                 coupling: float):
        super().__init__(name="Majorana fermion",
                         tex_name=r"\chi",
                         tex_decay = r'\chi \overline{\chi}',
                         mathematica_name=r"[\Chi]",
                         mass=mass,
                         coupling=coupling)

    def decay_width(self, ma: float, fa: float) -> float:
        # Decay width for Majorana DM candidate
        return np.abs(self.coupling)**2 * self.mass**2 * ma / (8 * np.pi * fa**2) * np.sqrt(1 - (2*ma/self.mass)**2)
    
    def __repr__(self) -> str:
        return f"DMMajorana(mass={self.mass}, coupling={self.coupling})"
    
    def _repr_markdown_(self):
        return f"**Dark Matter Majorana fermion**:\n\tmass = {self.mass}\n\tcoupling = {self.coupling}"
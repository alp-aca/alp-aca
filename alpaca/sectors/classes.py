import yaml
from ..decays.decays import to_tex
import os

class Sector:
    """
    A class representing a sector of observables.

    Attributes
    ----------
    name : str
        The name of the sector.
    observables : set
        A set of observables associated with the sector.
    tex : str
        The LaTeX representation of the sector name.
    description : str, optional
        A description of the sector (default is an empty string).
    """
    def __init__(self, name: str, observables: set, tex: str, description: str = ""):
        self.name = name
        self.observables = set(observables)
        self.tex = tex
        self.description = description

    def save(self, filename: str):
        """
        Save the sector to a YAML file.

        Parameters
        ----------
        filename : str
            The name of the file to save the sector to.
        """
        d = {'name': self.name, 'tex': self.tex, 'description': self.description, 'observables': list(self.observables)}
        with open(filename, 'w') as file:
            yaml.safe_dump(d, file)

    @classmethod
    def load(cls, filename: str):
        """
        Load a sector from a YAML file.

        Parameters
        ----------
        filename : str
            The name of the file to load the sector from.

        Returns
        -------
        Sector
            An instance of the Sector class.
        """
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
            data = {'description': ''} | data
            return cls(name=data['name'], tex=data['tex'], description=data['description'], observables=set(data['observables']))
        
    def _repr_markdown_(self):
        md = f"## {self.name}\n\n"
        md += f"**LaTeX**: {self.tex}\n\n"
        if self.description != "":
            md += f"**Description**: {self.description}\n\n"
        md += f"**Observables**:"
        for obs in self.observables:
            md += f"\n- {to_tex(obs)}"
        return md
        
def combine_sectors(sectors: list[Sector], name: str, tex: str, description: str = "") -> Sector:
    """
    Combine multiple sectors into a single sector.

    Parameters
    ----------
    sectors : list[Sector]
        A list of Sector instances to combine.
    name : str
        The name of the combined sector.
    tex : str
        The LaTeX representation of the combined sector name.
    description : str, optional
        A description of the combined sector (default is an empty string).

    Returns
    -------
    Sector
        An instance of the Sector class representing the combined sector.
    """
    observables = set()
    for sector in sectors:
        observables.update(sector.observables)
    
    return Sector(name=name, observables=observables, tex=tex, description=description)

def initialize_sectors(sector_dir: str|None = None) -> dict[str, Sector]:
    """
    Initialize sectors from YAML files in a directory.

    Parameters
    ----------
    sector_dir : str
        The directory containing the YAML files for the sectors.

    Returns
    -------
    dict[str, Sector]
        A dictionary mapping sector names to Sector instances.
    """
    if sector_dir is None:
        sector_dir = os.path.dirname(__file__)
    sectors = {}
    for filename in os.listdir(sector_dir):
        if filename.endswith('.yaml'):
            sector = Sector.load(os.path.join(sector_dir, filename))
            sectors[sector.name] = sector
    return sectors

default_sectors = initialize_sectors()
default_sectors['all'] = combine_sectors(list(default_sectors.values()), name='all', tex='Total', description='All sectors combined')
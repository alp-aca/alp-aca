import yaml
from ..decays.decays import to_tex, canonical_transition
import os

class Sector:
    """
    A class representing a sector of observables.

    Attributes
    ----------
    name : str
        The name of the sector.
    observables : set
        A set of observables associated with the sector. All measurements of these observables are included in the sector.
    obs_measurements : dict
        A dictionary of specific measurements for each observable in the sector.
    tex : str
        The LaTeX representation of the sector name.
    description : str, optional
        A description of the sector (default is an empty string).
    color : str, optional
        A color associated with the sector, used for plotting (default is None).
    """
    def __init__(self, name: str, tex: str, observables: set|None = None, obs_measurements: dict[str, set[str]] | None = None, description: str = "", color: str | None = None, lw: float | None = None, ls: str | None = None):
        self.name = name
        if observables is not None:
            self.observables = set()
            for obs in observables:
                if isinstance(obs, str):
                    self.observables.add(canonical_transition(obs))
                elif isinstance(obs, (list, tuple)):
                    self.observables.add((canonical_transition(obs[0]), obs[1]))
        else:
            self.observables = None
        if obs_measurements is not None:
            self.obs_measurements = {canonical_transition(k): set(v) for k, v in obs_measurements.items() if isinstance(k, str)}
            self.obs_measurements |= {(canonical_transition(k[0]), k[1]): set(v) for k, v in obs_measurements.items() if isinstance(k, (list, tuple))}
        else:
            self.obs_measurements = None
        self.tex = tex
        self.description = description
        self.color = color
        self.lw = lw
        self.ls = ls

    def save(self, filename: str):
        """
        Save the sector to a YAML file.

        Parameters
        ----------
        filename : str
            The name of the file to save the sector to.
        """
        d = {'name': self.name, 'tex': self.tex, 'description': self.description}
        if self.observables is not None:
            d |= {'observables': list(self.observables)}
        if self.obs_measurements is not None:
            d |= {'obs_measurements': self.obs_measurements}
        if self.color is not None:
            d |= {'color': self.color}
        if self.lw is not None:
            d |= {'lw': self.lw}
        if self.ls is not None:
            d |= {'ls': self.ls}

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
            if 'observables' not in data:
                data['observables'] = None
            if 'obs_measurements' not in data:
                data['obs_measurements'] = None
            if 'color' not in data:
                data['color'] = None
            if 'lw' not in data:
                data['lw'] = None
            if 'ls' not in data:
                data['ls'] = None
            return cls(name=data['name'], tex=data['tex'], description=data['description'], observables=data['observables'], obs_measurements=data['obs_measurements'], color=data['color'], lw=data['lw'], ls=data['ls'])

    def _repr_markdown_(self):
        md = f"## {self.name}\n\n"
        md += f"**LaTeX**: {self.tex}\n\n"
        if self.description != "":
            md += f"**Description**: {self.description}\n\n"
        md += f"**Observables**:"
        if self.observables is not None:
            for obs in self.observables:
                md += f"\n- {to_tex(obs)}"
        if self.obs_measurements is not None:
            for obs in self.obs_measurements.keys():
                md += f"\n- {to_tex(obs)}: {self.obs_measurements[obs]}"
        style = ''
        if self.color is not None:
            style += f"\n**Color**: <span style='color:{self.color}'>{self.color}</span>"
        if self.lw is not None:
            style += f"\n**Line Width**: {self.lw}"
        if self.ls is not None:
            style += f"\n**Line Style**: {self.ls}"
        if style:
            md += f"\n\n<details><summary>Plot style:</summary>\n{style}</details>"
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
        if sector.observables is not None:
            observables.update(sector.observables)
    obs_measurements = {}
    for sector in sectors:
        if sector.obs_measurements is not None:
            for obs in sector.obs_measurements.keys():
                if obs not in obs_measurements:
                    obs_measurements[obs] = set()
                obs_measurements[obs].update(sector.obs_measurements[obs])
    
    return Sector(name=name, observables=observables, tex=tex, description=description, obs_measurements=obs_measurements)

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
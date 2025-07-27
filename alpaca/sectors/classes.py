import yaml
from ..decays.decays import to_tex, canonical_transition
from ..experimental_data import get_measurements
import os
import copy

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
                self.observables.add(canonical_transition(obs))
        else:
            self.observables = None
        if obs_measurements is not None:
            if self.observables is not None:
                obs_measurements = {k: v for k, v in obs_measurements.items() if canonical_transition(k) not in self.observables}
            self.obs_measurements = {canonical_transition(k): set(v) for k, v in obs_measurements.items()}
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
    
    def contains_observable(self, observable: str) -> bool:
        """
        Check if the sector contains a specific observable.

        Parameters
        ----------
        observable : str
            The observable to check for.

        Returns
        -------
        bool
            True if the sector contains the observable, False otherwise.
        """
        res = False
        if self.observables is not None:
            if isinstance(observable, str):
                res = res or canonical_transition(observable) in self.observables
            elif isinstance(observable, (list, tuple)):
                res = res or (canonical_transition(observable[0]), observable[1]) in self.observables
        if self.obs_measurements is not None:
            if isinstance(observable, str):
                res = res or canonical_transition(observable) in self.obs_measurements.keys()
            elif isinstance(observable, (list, tuple)):
                res = res or (canonical_transition(observable[0]), observable[1]) in self.obs_measurements.keys()
        return res
    
    def exclude_observables(self, observables: str | list[str]) -> 'Sector':
        """
        Remove an observable from the sector.

        Parameters
        ----------
        observable : str
            The observable to remove.
        """
        if isinstance(observables, str):
            observables = [observables,]
        s = copy.copy(self)
        for observable in observables:
            if not s.contains_observable(observable):
                raise ValueError(f"Observable {observable} not found in sector {s.name}.")
            if s.observables is not None:
                s.observables.discard(canonical_transition(observable))
            if s.obs_measurements is not None:
                s.obs_measurements.pop(canonical_transition(observable), None)
        return s
    
    def exclude_measurements(self, measurements: tuple[str, str] | list[tuple[str, str]]) -> 'Sector':
        """
        Remove specific measurements from the sector.

        Parameters
        ----------
        measurements : tuple[str, str] | list[tuple[str, str]]
            The measurements to remove, specified as tuples of (observable, measurement).

        Returns
        -------
        Sector
            A new Sector instance with the specified measurements removed.
        """
        s = copy.copy(self)
        if isinstance(measurements, tuple):
            measurements = [measurements,]
        for measurement in measurements:
            observable, measurement_name = measurement
            if not s.contains_observable(observable):
                raise ValueError(f"Observable {observable} not found in sector {s.name}.")
            available_measurements = get_measurements(observable)
            if measurement_name not in available_measurements:
                raise ValueError(f"Measurement {measurement_name} not found for observable {observable} in sector {s.name}. Available measurements: {available_measurements}")
            if s.observables is not None and canonical_transition(observable) in s.observables:
                s.observables.remove(canonical_transition(observable))
                d_valid = {canonical_transition(observable): set(available_measurements) - {measurement_name}}
                if s.obs_measurements is None:
                    s.obs_measurements = d_valid
                else:
                    s.obs_measurements.update(d_valid)
            elif s.obs_measurements is not None and canonical_transition(observable) in s.obs_measurements.keys():
                if measurement_name in s.obs_measurements[canonical_transition(observable)]:
                    s.obs_measurements[canonical_transition(observable)].remove(measurement_name)
                    if not s.obs_measurements[canonical_transition(observable)]:
                        del s.obs_measurements[canonical_transition(observable)]
        return s


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
'''Dictionary of the pre-defined sectors in ALPaca.'''
default_sectors['all'] = combine_sectors(list(default_sectors.values()), name='all', tex='Total', description='All sectors combined')
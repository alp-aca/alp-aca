"""ALPaca: The ALP Automatic Computing Algorithm

Modules
-------
uvmodels
    Contains the classes to define the UV models.
experimental_data
    Contains classes and functions to handle experimental data.
sectors
    Contains classes and functions to handle sectors of observables.
statistics
    Contains functions to handle statistics.
plotting
    Contains functions to handle plotting.
citations
    Contains functions to handle bibliographical references.
scan
    Contains functions to handle parameter space scans.
benchmarks
    Contains the benchmarks defined in 1901.09966 [hep-ex] for ALPs.

Classes
-------
ALPcouplings
    A class to represent the couplings of ALPs to SM particles.
ALPcouplingsEncoder
    A class to encode ALPcouplings objects into JSON format.
ALPcouplingsDecoder
    A class to decode JSON formatted ALPcouplings objects.

Functions
---------
decay_width
    Calculates the decay width of a particle.
branching_ratio
    Calculates the branching ratio of a particle.
cross_section
    Calculates the cross section of a process.
meson_mixing
    Calculates the value of a meson mixing observable.
alp_channels_decay_widths
    Calculates the decay widths for all ALP channels.
alp_channels_branching_ratios
    Calculates the branching ratios for all ALP channels.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

__version__ = "1.1.0"

class info:
    def __repr__(self):
        out = "\nALPaca: ALP Automatic Computing Algorithm\n"
        out += "=========================================\n"
        out += f"Version: {__version__}\n\n"
        out += "Authors:\n"
        out += "\t- Jorge Alda (U. Padova & INFN Padova & CAPA Zaragoza)\n"
        out += "\t- Marta Fuentes Zamoro (U. Autónoma de Madrid & IFT Madrid)\n"
        out += "\t- Luca Merlo (U. Autónoma de Madrid & IFT Madrid)\n"
        out += "\t- Xavier Ponce Díaz (U. Basel)\n"
        out += "\t- Stefano Rigolin (U. Padova & INFN Padova)\n"
        out += "Homepage: https://github.com/alp-aca/alp-aca\n"
        out += "Documentation: https://alpaca-alps.readthedocs.io/latest/\n"
        out += "Please cite arXiv:2508.08354 https://arxiv.org/abs/2508.08354"
        return out
    def _repr_markdown_(self):
        import os
        import pathlib
        current_dir = os.path.dirname(__file__)
        readme_path = pathlib.Path(current_dir).parent / "README.md"
        with open(readme_path, "r") as f:
            readme_content = f.read()
        return readme_content

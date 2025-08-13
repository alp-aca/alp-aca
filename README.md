# ALPaca

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16447037.svg)](https://doi.org/10.5281/zenodo.16447037) [![arxiv](https://img.shields.io/badge/arXiv-2508.08354_[hep--ph]-B31B1B.svg?style=flat&logo=arxiv&logoColor=B31B1B)](https://arxiv.org/abs/2508.08354)

Welcome to the ALP Automated Computed Algorithm (ALPaca)!

![ALPaca logo](docs/_static/logo.png)

ALPaca is an open-source Python library for the phenomenology of Axion-Like Particles (ALPs) with masses in the ranges of $m_a \sim 0.01 - 10\,\mathrm{GeV}$, mainly in processes involving mesons.

ALPaca integrates the full analysis with an easy-to-use syntax:

* Matching of selected UV-complete models (DFSZ-like, KSVZ-like, flaxions, etc.) to the ALP-EFT.
* Numerical running and matching of the ALP-EFT coefficients down to the physical relevant scales, including ALP-$\chi\!$ PT.
* Calulation of decay rates for processes involving ALPs:
  * ALP production in rare meson decays $M_1\to M_2 a$, quarkonia decays $V\to \gamma a$ and non-resonant production $e^+e^- \to \gamma a$,
  * ALP decays into photons, leptons and mesons,
  * Processes mediated by on-shell ALPs in the Narrow Width Approximation,
  * Leptonic and radiative meson decays, and meson mixing, with off-shell ALPs.
* Calculation of ALP decay lengths and probability of decaying outside the detector, with a displaced vertex or in the prompt region.
* $\chi^2$ statistical analysis, with fine-grained control of the observables and experimental measurements included.
* Generation of publication-grade exclusion plots.
* Automatic management of the bibliographical references used in the analysis.

## The ALPaca team

* **Jorge Alda**: Università degli Studi di Padova & INFN Sezione di Padova & CAPA Zaragoza.
* **Marta Fuentes Zamoro**: Universidad Autónoma de Madrid & IFT Madrid.
* **Luca Merlo**: Universidad Autónoma de Madrid & IFT Madrid.
* **Xavier Ponce Díaz**: University of Basel.
* **Stefano Rigolin**: Università degli Studi di Padova & INFN Sezione di Padova.

## ALPaca in action

In [this repositoy](https://github.com/alp-aca/examples) you can find examples, tutorials and applications of ALPaca.

ALPaca has been used in the following publications:

* J. Alda, M. Fuentes Zamoro, L. Merlo, X. Ponce Díaz, S. Rigolin: *Comprehensive ALP searches in Meson Decays*. [arXiv:2507.19578](https://arxiv.org/abs/2507.19578)

If you have used ALPaca in your publication and want to be featured in this list, please [contact us](https://github.com/alp-aca/alp-aca/issues/new?template=publication-using-alpaca.md).

## Installation

ALPaca can be installed with `pip`:

```bash
pip3 install alpaca-ALPs
```

It is *strongly recommended* to install ALPaca inside a virtual environment (venv), in order to avoid clashes with conflicting versions of the dependencies. In order to create a venv, execute the following command

```bash
python3 -m venv pathToVenv
```

where `pathToVenv` is the location where the files of the venv will be stored. In order to activate the venv, for Linux or MacOS using `bash` or `zsh`

```bash
source pathToVenv/bin/activate
```

For Windows using ```cmd.exe```

```bat
C:\> pathToVenv\Scripts\Activate.bat
```

And for Windows using ```PowerShell```

```powershell
PS C:\> path_to_venv\Scripts\Activate.ps1
```

Once the venv is activated, ALPaca can be normally installed and used.

## Citing ALPaca

If you use ALPaca, please cite

```bibtex
@article{Alda:2025nsz,
    author = "Alda, Jorge and Fuentes Zamoro, Marta and Merlo, Luca and Ponce D{\'\i}az, Xavier and Rigolin, Stefano",
    title = "{ALPaca: The ALP Automatic Computing Algorithm}",
    eprint = "2508.08354",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "IFT-UAM/CSIC-25-82",
    month = "8",
    year = "2025"
}

@software{alda_2025_16447037,
  author       = {Alda, Jorge and
                  Fuentes Zamoro, Marta and
                  Merlo, Luca and
                  Rigolin, Stefano and
                  Ponce Díaz, Xavier},
  title        = {ALPaca v1.0},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0-alpha.1},
  doi          = {10.5281/zenodo.16447037},
  url          = {https://doi.org/10.5281/zenodo.16447037},
}
```

## Documentation

The ALPaca manual is available on [arXiv](https://arxiv.org/abs/2508.08354).

You can also check the [automatically-generated documentation](https://alpaca-alps.readthedocs.io/latest/).

## Feedback

If you encounter bugs or want to propose a new feature, you can contact us using [Gihub issues](https://github.com/alp-aca/alp-aca/issues/new/choose).

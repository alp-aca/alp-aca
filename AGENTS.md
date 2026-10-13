# AGENTS.md

## Project overview

ALP-aca is a scientific Python package for axion-like-particle phenomenology.
Treat numerical conventions, physical dimensions, coupling normalisations,
kinematic domains, and literature inputs as part of the public behaviour.

## Environment and dependencies

- Supported Python versions are declared in `pyproject.toml` (currently 3.11--3.14).
- Use Poetry and the committed `poetry.lock` for development environments.
- Prefer `poetry install` and `poetry run <command>` when no active project
  environment is already available.
- Plotting support is optional and is split into the `matplotlib` and `plotly`
  extras in `pyproject.toml`.
- Do not add dependencies or modify `poetry.lock` unless the requested change
  requires it.
- Do not install packages or access the network when the existing environment
  is sufficient.

## Repository map

- `alpaca/`: importable package source.
- `alpaca/classes.py`: core coupling classes.
- `alpaca/constants.py` and `alpaca/common.py`: shared constants and utilities.
- `alpaca/decays/`: decay rates, branching ratios, effective couplings, and
  particle definitions. Baryonic processes live under `alpaca/decays/baryons/`.
- `alpaca/chiPT/`: chiral-perturbation-theory calculations and form factors.
- `alpaca/rge/`: running and matching of couplings.
- `alpaca/experimental_data/`: measurements and numerical experimental inputs.
- `alpaca/sectors/`: sector definitions and YAML configuration.
- `alpaca/plotting/`: Matplotlib and Plotly interfaces.
- `alpaca/statistics/` and `alpaca/scan/`: statistical and parameter-scan tools.
- `alpaca/uvmodels/`: ultraviolet-model definitions and matching.
- `docs/`: Sphinx documentation written primarily in reStructuredText.
- `var/`: exploratory notebooks, scripts, and local analysis artifacts; it is
  ignored by Git and is not package source.

## Working practices

- Start from the files named by the user, then inspect only their direct callers,
  dependencies, and nearby conventions before broadening the search.
- Make focused changes and avoid unrelated refactors, formatting churn, or
  reordering imports without a concrete reason.
- Preserve unrelated modified and untracked files. Never discard user changes.
- Keep public interfaces backward compatible unless an interface change is
  explicitly requested.
- When a public Python interface changes, update the corresponding `.pyi` stub
  when one exists.
- Reuse existing abstractions, constants, particle definitions, and bibliography
  helpers instead of duplicating them.
- Keep optional plotting dependencies out of core import paths.

## Scientific integrity

- Preserve units and state the expected dimensions of new physical quantities.
- Preserve established coupling conventions and normalisations. Trace them to
  their definitions before changing a formula.
- Enforce physical domains explicitly, including masses, thresholds, and allowed
  decay channels.
- Do not silently change numerical constants, benchmark values, form-factor
  inputs, experimental data, or literature-derived assumptions.
- Distinguish exact expressions from approximations and document the regime in
  which an approximation is valid.
- For substantive formula changes, explain the derivation or source and check
  limiting cases when practical.
- Treat `.npy`, `.npz`, `.pickle`, YAML, CSV, and text data files as curated
  inputs. Do not regenerate or edit them unless the task explicitly requires it,
  and record their provenance when adding new data.

## Notebooks and generated artifacts

- Keep reusable scientific logic in `alpaca/`; use notebooks in `var/` mainly
  for exploration, demonstrations, and plots.
- Avoid notebook-wide rewrites when changing a cell. Preserve unrelated cells,
  metadata, and outputs unless the user requests otherwise.
- Do not claim that notebook outputs or figures were regenerated unless the
  relevant cells actually ran successfully.
- Do not commit generated documentation under `docs/_build/` or distribution
  artifacts under `dist/`.

## Documentation and reporting

- Update docstrings and files under `docs/` when a requested change alters public
  behaviour or user-facing APIs.
- Preserve the existing documentation format and naming conventions.
- At completion, summarise the changed files, the scientific or API impact, the
  commands actually run, and any validation that could not be completed.

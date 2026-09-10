[![DOI](https://zenodo.org/badge/DOI/10.1038/s41587-022-01489-7.svg)](https://doi.org/10.1038/s41587-022-01489-7)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7009614.svg)](https://doi.org/10.5281/zenodo.7009614)

# rotational_diffusion_photophysics
Tools for computing time-dependent fluorescence signals with rotational diffusion and complex photophysics.  
An arbitrary kinetic scheme can be used and the program analytically solves the diffusion-kinetics problem. All the angular probability distribution functions are expanded with spherical harmonics.  
This repository was used in the scientific paper [A. Volpato et al. Nat Biotechnol 41, 552–559 (2023)](https://doi.org/10.1038/s41587-022-01489-7).

## Module content
The pacakge is usually imported as `import rotational_diffusion_photophyscis as rdp`

- `rdp.engine`  
Main engine of the kinetics and diffusion solver for an arbitrary kinetics scheme and isotorpic free rotatioal diffusion.

- `rdp.models`  
Several models to create a full system class `rdp.engine.System`:
  - `rdp.models.illumination`  
  Classes with laser modulation
  - `rdp.models.detection`  
  Classes for detection. Currently only `PolarizedDetection`
  - `rdp.models.diffusion`  
  Classes with rotational diffusion models
  - `rdp.models.fluorophore`  
  Fluorophore models, including `NegativeSwitcher` and `STEDDye`
  - `rdp.models.starss`  
  Classes for STARSS experiments. A few `System` classes are defined.

- `rdp.plot`  
Some useful plotting tools for orientational probabilities and STARSS pulse schemes.


## Additional content

- `starss` folder  
Notebooks with computation and plotting of example STARSS simulations.

- `notes` folder  
Mathematical notes about the rotational diffusion and kinetics model.


## Experiments and the run store

An experiment is a frozen parameter dataclass with a `run()` method, one module per
experiment in `rdp.experiments` (`exp001_starss_method1`, `exp010_steady_state`, …).
The runners in `scripts/experiments/sNNN_*.py` import one, run it, and plot it.

Every run is **stored outside git**, so simulations are not lost and a repeated run
costs nothing:

```python
from rotational_diffusion_photophysics import store
from rotational_diffusion_photophysics.experiments.exp010_steady_state import experiment

params = experiment(tau=100e-6)
res = store.run_cached(params, label='tau-100us')   # runs once, then loads
store.save_open_figures(store.run_dir(params))       # figures next to the data
```

One folder per run, named `<date>_<label>_<hash>`, holding `params.json`,
`arrays.npz`, `scalars.json`, `meta.json` (git commit, dirty flag and the diff of a
dirty tree, versions, runtime), `figures/` and an optional `note.md`. The folder is
addressed by the sha256 of the parameters — including a fingerprint of the
fluorophore's numbers, so an edited preset never reuses an old run's identity.
Figures that belong to a *set* of runs (a sweep, a comparison) go to
`store.session(name)` with a `runs.txt` naming the runs behind them.

Set the root with the `RDP_RUNS` environment variable, pointing at a synced or
backed-up directory:

```powershell
$env:RDP_RUNS = "G:\My Drive\...\rdp-runs"     # persistent: set it in the user env
```

Without it the store falls back to `<repo>/runs`, which is git-ignored — that works,
but it is not a backup. Nothing inside a run records an absolute path, so the store
can be moved later by moving the folders.

Query what has been simulated with `store.find('exp010_steady_state', tau=100e-6)`
(params, observables and provenance per run) or dump a one-line-per-run index with
`store.write_index(path)`. Keep that index local: it is derived and can be rebuilt
from the run folders at any time.

## Install
Using pip and making a symlink.
From the main folder launch: `pip install -e .`
"""Run store: persist experiment parameters and outputs outside git.

The store is *parameter-addressed*: the sha256 of a frozen parameter dataclass
is the identity of a run, so the same parameters mean the same folder, and a
cache hit instead of a recomputation.

    $RDP_RUNS/exp010_steady_state/2026-09-07_power-sweep_a3f19c2b/
        arrays.npz  scalars.json  meta.json  params.json  figures/  note.md
    $RDP_RUNS/_sessions/2026-09-07_120000_s010/   output spanning several runs

``params.json`` is written LAST, so a folder is a complete run if and only if
it has one; a half-written folder is invisible to :func:`load` and :func:`find`.

The root is the ``root`` argument, else ``$RDP_RUNS``, else ``<repo>/runs``
(git-ignored). No absolute path is recorded inside a run, so moving the store
is just moving the folders. Provenance is recorded, never enforced: a run made
on a dirty tree is saved like any other, with its diff alongside.

    res = store.run_cached(params, label='baseline')
    store.save_open_figures(store.session('s010'))   # before plt.show()
"""
import dataclasses
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

__all__ = [
    "runs_root", "params_dict", "params_hash", "experiment_name",
    "save", "load", "run_cached", "run_dir", "find", "write_index",
    "session", "save_open_figures", "save_figures",
]

SCHEMA_VERSION = 1
_DIFF_LIMIT = 256 * 1024          # cap the stored diff of a dirty tree [bytes]
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SCALARS = (bool, int, float, str, type(None))
_NP_SCALARS = (np.integer, np.floating, np.bool_)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _p(path) -> Path:
    r"""Long-path-safe path: past 240 characters Windows needs the ``\\?\`` prefix.

    The store nests and can live under a long cloud-drive path. Without this,
    pathlib quietly reports a deep folder as empty instead of failing.
    """
    text = str(Path(path).resolve())
    if os.name == "nt" and len(text) > 240 and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\" + text)
    return Path(text)


def _read(path) -> str:
    return _p(path).read_text(encoding="utf-8")


def _write(path, text):
    _p(path).write_text(text, encoding="utf-8")


def _read_json(path) -> dict:
    """Parsed json, or ``{}`` if the file is missing or half-written."""
    try:
        return json.loads(_read(path))
    except (ValueError, OSError):
        return {}


def _mkdir(path) -> Path:
    _p(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def _listdir(path):
    """Children of a directory, sorted; empty if it does not exist."""
    directory = _p(path)
    return [Path(path) / c.name for c in sorted(directory.iterdir())] \
        if directory.is_dir() else []


def _slug(text) -> str:
    return _SAFE.sub("-", str(text)).strip("-")


def _repo_root() -> Path:
    """The repository containing this package, or the current directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def runs_root(root=None) -> Path:
    """Root of the store: ``root``, else ``$RDP_RUNS``, else ``<repo>/runs``."""
    if root is not None:
        path = Path(root)
    elif os.environ.get("RDP_RUNS"):
        path = Path(os.environ["RDP_RUNS"]).expanduser()
    else:
        path = _repo_root() / "runs"
    return _mkdir(path)


# ---------------------------------------------------------------------------
# Parameter identity
# ---------------------------------------------------------------------------
def experiment_name(params) -> str:
    """Name of the experiment: the module its parameter dataclass comes from."""
    return type(params).__module__.rsplit(".", 1)[-1]


def _fingerprint(obj) -> str:
    """sha256 over the numeric attributes of a model object (a fluorophore).

    Presets are module-level instances with no ``__eq__``, so the name alone
    cannot tell that a cross-section was edited. This can: an edited preset
    fingerprints differently and lands in a different folder.
    """
    h = hashlib.sha256(type(obj).__name__.encode())
    for name, value in sorted(vars(obj).items()):
        if name.startswith("_"):
            continue
        h.update(name.encode())
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            h.update(f"{arr.dtype}{arr.shape}".encode())
            h.update(arr.tobytes())
        elif isinstance(value, _SCALARS):
            h.update(repr(value).encode())
        elif isinstance(value, (list, tuple)):
            h.update(repr(list(value)).encode())
    return h.hexdigest()


def _jsonable(value):
    """One parameter value, JSON-ready.

    Anything that is not a scalar, sequence or array is a model object and
    becomes a ``{name, class, fingerprint}`` descriptor. ``name`` is found by
    identity against the presets, so ``rsEGFP2_8states`` is recorded as such,
    and an object built inline is recorded as ``None``.
    """
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, _NP_SCALARS):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    from rotational_diffusion_photophysics.models import fluorophore as _fl
    return {"name": next((k for k, v in vars(_fl).items() if v is value), None),
            "class": type(value).__name__, "fingerprint": _fingerprint(value)}


def params_dict(params) -> dict:
    """Every field of a parameter dataclass, JSON-ready."""
    return {f.name: _jsonable(getattr(params, f.name))
            for f in dataclasses.fields(params)}


def params_hash(params) -> str:
    """sha256 of the experiment name plus its parameters.

    Adding a field changes the hash, so the cache misses after a schema change;
    runs stored earlier stay readable and findable, they just stop being hits.
    """
    payload = {"experiment": experiment_name(params), "schema": SCHEMA_VERSION,
               "params": params_dict(params)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _git(*args, cwd) -> str:
    try:
        out = subprocess.run(("git",) + args, cwd=str(cwd), capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def _provenance() -> tuple:
    """``(meta, diff)`` for the package repository.

    A dirty tree is recorded, not refused -- with the diff of its tracked files
    so the run stays reproducible. The diff is truncated past ``_DIFF_LIMIT``.
    """
    repo = _repo_root()
    status = _git("status", "--porcelain", cwd=repo)
    diff = _git("diff", "HEAD", cwd=repo) if status.strip() else ""
    truncated = len(diff.encode()) > _DIFF_LIMIT
    if truncated:
        diff = diff.encode()[:_DIFF_LIMIT].decode(errors="ignore")
    try:
        from importlib.metadata import version
        package_version = version("rotational-diffusion-photophysics")
    except Exception:
        package_version = ""
    meta = {
        "git_commit": _git("rev-parse", "HEAD", cwd=repo).strip() or None,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo).strip() or None,
        "git_dirty": bool(status.strip()),
        "git_dirty_files": [line[3:] for line in status.splitlines()],
        "diff_truncated": truncated,
        "package_version": package_version,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "host": platform.node(),
    }
    return meta, diff


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def _split_results(res) -> tuple:
    """A results dataclass as ``(arrays, scalars, skipped)``.

    ``skipped`` names the fields that are neither -- the live engine
    ``system``, derived from the parameters and rebuilt by :func:`load`.
    """
    arrays, scalars, skipped = {}, {}, []
    for f in dataclasses.fields(res):
        value = getattr(res, f.name)
        if isinstance(value, np.ndarray):
            arrays[f.name] = value
        elif isinstance(value, _SCALARS):
            scalars[f.name] = value
        elif isinstance(value, _NP_SCALARS):
            scalars[f.name] = value.item()
        elif isinstance(value, (tuple, list)) and \
                all(isinstance(v, _SCALARS) for v in value):
            scalars[f.name] = list(value)
        else:
            skipped.append(f.name)
    return arrays, scalars, skipped


def run_dir(params, root=None):
    """The stored folder for these parameters, or ``None`` if there is none."""
    tag = params_hash(params)
    for folder in _listdir(runs_root(root) / experiment_name(params)):
        if folder.name.endswith(f"_{tag[:8]}") and _read_json(
                folder / "params.json").get("_meta", {}).get("params_hash") == tag:
            return folder
    return None


def save(params, res, label=None, note=None, figures=None, root=None,
         runtime=None):
    """Store one run and return its folder.

    ``label`` is a short tag for the folder name, ``note`` becomes ``note.md``,
    ``figures`` is ``{name: matplotlib figure}``, ``runtime`` is the wall-clock
    seconds of the run. Re-saving the same parameters reuses the existing
    folder, so a script can be re-run without duplicating anything.
    """
    tag = params_hash(params)
    target = run_dir(params, root)
    if target is None:
        name = "_".join(filter(None, (datetime.now().strftime("%Y-%m-%d"),
                                      _slug(label) if label else None, tag[:8])))
        target = runs_root(root) / experiment_name(params) / name
    _mkdir(target)

    arrays, scalars, skipped = _split_results(res)
    scalars["_skipped_fields"] = skipped
    with open(_p(target / "arrays.npz"), "wb") as handle:
        np.savez_compressed(handle, **arrays)
    _write(target / "scalars.json", json.dumps(scalars, indent=2, sort_keys=True))

    meta, diff = _provenance()
    meta.update({"saved": datetime.now().astimezone().isoformat(timespec="seconds"),
                 "runtime_s": runtime,
                 "results_class": type(res).__qualname__,
                 "arrays": {k: list(v.shape) for k, v in arrays.items()}})
    _write(target / "meta.json", json.dumps(meta, indent=2, sort_keys=True))
    if diff:
        _write(target / "dirty.diff", diff)
    if note:
        _write(target / "note.md", str(note))
    if figures:
        save_figures(target, figures)

    # params.json marks the folder complete, so it is written last.
    _write(target / "params.json", json.dumps(
        {"_meta": {"schema": SCHEMA_VERSION, "experiment": experiment_name(params),
                   "params_class": type(params).__qualname__,
                   "params_hash": tag, "label": label},
         "params": params_dict(params)}, indent=2, sort_keys=True))
    return target


def load(params, root=None, rebuild_system=True):
    """Return the stored results for these parameters, or ``None``.

    A ``system`` field cannot be stored -- it is the live engine -- so it is
    rebuilt with ``params.build()``. That system is **fresh, not solved**: it
    carries the pulse scheme but none of the state a completed run leaves.
    """
    target = run_dir(params, root)
    if target is None:
        return None

    scalars = _read_json(target / "scalars.json")
    scalars.pop("_skipped_fields", None)
    with np.load(_p(target / "arrays.npz")) as npz:
        arrays = {k: npz[k] for k in npz.files}

    results = sys.modules[type(params).__module__].results
    kwargs = {}
    for f in dataclasses.fields(results):
        if f.name in arrays:
            kwargs[f.name] = arrays[f.name]
        elif f.name in scalars:
            kwargs[f.name] = scalars[f.name]
        elif f.name == "system":
            kwargs[f.name] = params.build() if rebuild_system else None
        else:
            kwargs[f.name] = None
    return results(**kwargs)


def run_cached(params, root=None, **save_kwargs):
    """Load this run if it is stored, otherwise run it and store it."""
    cached = load(params, root)
    if cached is not None:
        return cached
    t0 = time.perf_counter()
    res = params.run()
    save(params, res, root=root, runtime=time.perf_counter() - t0, **save_kwargs)
    return res


# ---------------------------------------------------------------------------
# Figures and sessions
# ---------------------------------------------------------------------------
def save_figures(target, figures):
    """Write ``{name: figure}`` as PNGs under ``<target>/figures``."""
    directory = _mkdir(Path(target) / "figures")
    for name, fig in figures.items():
        fig.savefig(_p(directory / f"{_slug(name)}.png"), dpi=150,
                    bbox_inches="tight")
    return directory


def save_open_figures(target, prefix=None):
    """Write every open matplotlib figure under ``<target>/figures``.

    Call it before ``plt.show()``, so a runner does not have to keep references
    to the figures it drew.
    """
    import matplotlib.pyplot as plt

    figures = {}
    for num in plt.get_fignums():
        fig = plt.figure(num)
        stem = fig.get_suptitle() if hasattr(fig, "get_suptitle") else ""
        stem = stem or (fig.axes[0].get_title() if fig.axes else "")
        stem = _slug(stem)[:40] or f"figure{num}"
        figures[f"{prefix + '-' if prefix else ''}{num:02d}-{stem}"] = fig
    return save_figures(target, figures) if figures else None


def session(name, root=None):
    """A folder for output belonging to a *set* of runs, not to one run.

    Sweeps and comparison figures span many runs, so they belong in no single
    run folder. Write them here and list the runs they came from::

        sess = store.session('s010_steady_state_starss')
        store.save_open_figures(sess)
        (sess / 'runs.txt').write_text('\\n'.join(str(p) for p in run_paths))
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return _mkdir(runs_root(root) / "_sessions" / f"{stamp}_{_slug(name)}")


# ---------------------------------------------------------------------------
# Querying the store
# ---------------------------------------------------------------------------
def find(experiment=None, root=None, **filters):
    """Every stored run, newest first, optionally filtered by parameter value.

    ``find('exp010_steady_state', tau=30e9)`` returns the runs of that
    experiment with that correlation time. Each hit is a dict with ``path``,
    ``experiment``, ``hash``, ``label``, ``params``, ``scalars`` and ``meta``.
    """
    base = runs_root(root)
    directories = [base / experiment] if experiment else \
        [d for d in _listdir(base) if d.name != "_sessions"]

    hits = []
    for directory in directories:
        for folder in _listdir(directory):
            record = _read_json(folder / "params.json")
            params = record.get("params", {})
            if not record or any(params.get(k) != v for k, v in filters.items()):
                continue                      # incomplete folder, or filtered out
            meta = record["_meta"]
            hits.append({"path": folder, "experiment": meta["experiment"],
                         "hash": meta["params_hash"], "label": meta.get("label"),
                         "params": params,
                         "scalars": _read_json(folder / "scalars.json"),
                         "meta": _read_json(folder / "meta.json")})
    hits.sort(key=lambda h: h["meta"].get("saved", ""), reverse=True)
    return hits


def write_index(path, experiment=None, root=None) -> Path:
    """Write one JSON line per stored run to ``path``.

    The index is *derived*: delete it and rewrite it from the run folders at
    any time, so keep it local and out of the synced store (where a shared
    append-only file would collect conflict copies).
    """
    _mkdir(Path(path).parent)
    _write(path, "".join(json.dumps({
        "hash": hit["hash"][:12],
        "experiment": hit["experiment"],
        "label": hit["label"],
        "saved": hit["meta"].get("saved"),
        "git_commit": (hit["meta"].get("git_commit") or "")[:12],
        "git_dirty": hit["meta"].get("git_dirty"),
        "path": str(hit["path"]),
        "params": hit["params"],
        "scalars": {k: v for k, v in hit["scalars"].items()
                    if not k.startswith("_")},
    }, sort_keys=True) + "\n" for hit in find(experiment, root)))
    return Path(path)

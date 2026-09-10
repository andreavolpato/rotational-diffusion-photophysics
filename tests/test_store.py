"""Tests for the run store: parameter identity, round-trip, and querying.

Runs are made deliberately cheap (few time points, small ``lmax``) -- what is
under test is the storage contract, not the physics.
"""
from dataclasses import replace

import json

import numpy as np
import pytest

from rotational_diffusion_photophysics import store
from rotational_diffusion_photophysics.experiments.exp001_starss_method1 import (
    experiment as exp001)
from rotational_diffusion_photophysics.experiments.exp010_steady_state import (
    experiment as exp010)


def _cheap(params):
    """A fast variant of an experiment: coarse time grid, small basis."""
    return replace(params, n_time=8, lmax=2)


@pytest.fixture
def cheap_run():
    params = _cheap(exp001())
    return params, params.run()


# ---------------------------------------------------------------------------
# Parameter identity
# ---------------------------------------------------------------------------
def test_experiment_name_is_the_module():
    assert store.experiment_name(exp001()) == "exp001_starss_method1"
    assert store.experiment_name(exp010()) == "exp010_steady_state"


def test_params_dict_is_json_serializable():
    payload = store.params_dict(exp010())
    json.dumps(payload)                       # must not raise
    assert payload["fluorophore"]["name"] == "rsEGFP2_8states"
    assert len(payload["fluorophore"]["fingerprint"]) == 64


def test_hash_is_stable_and_parameter_sensitive():
    params = exp001()
    assert store.params_hash(params) == store.params_hash(exp001())
    assert store.params_hash(params) != store.params_hash(replace(params, tau=1e-3))
    # different experiments with identical field values stay distinct
    assert store.params_hash(exp001()) != store.params_hash(exp010())


def test_hash_follows_an_edited_fluorophore_preset():
    """An edited preset must not silently reuse the old run's identity."""
    from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states

    params = exp001()
    before = store.params_hash(params)
    original = rsEGFP2_8states.lifetime_on
    try:
        rsEGFP2_8states.lifetime_on = original * 2
        assert store.params_hash(params) != before
    finally:
        rsEGFP2_8states.lifetime_on = original
    assert store.params_hash(params) == before


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def test_save_writes_a_complete_run(tmp_path, cheap_run):
    params, res = cheap_run
    target = store.save(params, res, label="unit test", note="why not",
                        root=tmp_path)

    assert target.parent.name == "exp001_starss_method1"
    assert target.name.endswith(store.params_hash(params)[:8])
    assert "unit-test" in target.name
    for name in ("params.json", "scalars.json", "meta.json", "arrays.npz",
                 "note.md"):
        assert (target / name).is_file()

    manifest = json.loads((target / "params.json").read_text())
    assert manifest["_meta"]["params_hash"] == store.params_hash(params)
    assert manifest["params"]["tau"] == params.tau

    meta = json.loads((target / "meta.json").read_text())
    assert meta["results_class"] == "results"
    assert set(meta["arrays"]) == {"t", "signals", "anisotropy"}
    assert "git_dirty" in meta                # provenance recorded, not enforced


def test_round_trip_returns_the_same_arrays(tmp_path, cheap_run):
    params, res = cheap_run
    store.save(params, res, root=tmp_path)

    loaded = store.load(params, root=tmp_path)
    assert loaded is not None
    np.testing.assert_allclose(loaded.t, res.t)
    np.testing.assert_allclose(loaded.signals, res.signals)
    np.testing.assert_allclose(loaded.anisotropy, res.anisotropy)


def test_round_trip_of_a_results_class_with_scalars(tmp_path):
    params = _cheap(exp010())
    res = params.run()
    store.save(params, res, root=tmp_path)

    loaded = store.load(params, root=tmp_path)
    assert loaded.anisotropy == pytest.approx(res.anisotropy)
    assert loaded.polarization_snr == pytest.approx(res.polarization_snr)
    assert loaded.total_emitted_photons == pytest.approx(res.total_emitted_photons)
    np.testing.assert_array_equal(loaded.channel_window, res.channel_window)


def test_system_is_rebuilt_unsolved(tmp_path, cheap_run):
    """``system`` is derived, so it is rebuilt from the parameters on load."""
    params, res = cheap_run
    store.save(params, res, root=tmp_path)

    loaded = store.load(params, root=tmp_path)
    assert type(loaded.system) is type(res.system)
    assert store.load(params, root=tmp_path, rebuild_system=False).system is None


def test_load_misses_on_different_parameters(tmp_path, cheap_run):
    params, res = cheap_run
    store.save(params, res, root=tmp_path)
    assert store.load(replace(params, tau=7e-6), root=tmp_path) is None


def test_incomplete_folder_is_invisible(tmp_path, cheap_run):
    params, res = cheap_run
    target = store.save(params, res, root=tmp_path)
    (target / "params.json").unlink()         # simulate an interrupted write
    assert store.load(params, root=tmp_path) is None
    assert store.find(root=tmp_path) == []


def test_resaving_reuses_the_folder(tmp_path, cheap_run):
    params, res = cheap_run
    first = store.save(params, res, root=tmp_path)
    second = store.save(params, res, root=tmp_path)
    assert first == second
    assert len(list(first.parent.iterdir())) == 1


def test_run_cached_runs_once(tmp_path, monkeypatch):
    params = _cheap(exp001())
    calls = []
    original = type(params).run

    def counting_run(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(type(params), "run", counting_run)
    first = store.run_cached(params, root=tmp_path)
    second = store.run_cached(params, root=tmp_path)
    assert len(calls) == 1
    np.testing.assert_allclose(first.signals, second.signals)


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------
def test_find_filters_by_parameter(tmp_path):
    fast = _cheap(exp001())
    slow = replace(fast, tau=1e-3)
    for params in (fast, slow):
        store.save(params, params.run(), root=tmp_path)

    assert len(store.find(root=tmp_path)) == 2
    hits = store.find("exp001_starss_method1", root=tmp_path, tau=1e-3)
    assert len(hits) == 1
    assert hits[0]["params"]["tau"] == 1e-3
    assert hits[0]["scalars"]["_skipped_fields"] == ["system"]
    assert store.find("exp010_steady_state", root=tmp_path) == []


def test_write_index_lists_every_run(tmp_path, cheap_run):
    params, res = cheap_run
    store.save(params, res, label="indexed", root=tmp_path)

    index = store.write_index(tmp_path / "local" / "index.jsonl", root=tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["label"] == "indexed"
    assert rows[0]["experiment"] == "exp001_starss_method1"
    assert rows[0]["params"]["tau"] == params.tau


# ---------------------------------------------------------------------------
# Root resolution and sessions
# ---------------------------------------------------------------------------
def test_root_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RDP_RUNS", str(tmp_path / "drive" / "rdp-runs"))
    assert store.runs_root() == tmp_path / "drive" / "rdp-runs"
    assert store.runs_root().is_dir()


def test_root_falls_back_to_the_repository(monkeypatch):
    monkeypatch.delenv("RDP_RUNS", raising=False)
    assert store.runs_root().name == "runs"


def test_no_absolute_path_is_recorded_in_a_run(tmp_path, cheap_run):
    """The store must be movable: nothing inside a run may pin its location."""
    params, res = cheap_run
    target = store.save(params, res, root=tmp_path)
    for name in ("params.json", "scalars.json", "meta.json"):
        assert str(tmp_path) not in (target / name).read_text()


def test_deep_store_still_writes(tmp_path, cheap_run):
    """The store nests, and a cloud-drive root is long: 260 characters is not
    a limit the store may fail at (it did, on the first figure written into a
    session folder)."""
    deep = tmp_path
    while len(str(deep)) < 200:
        deep = deep / ("nested-directory-with-a-long-name")
    params, res = cheap_run
    target = store.save(params, res, label="a-deliberately-long-run-label",
                        root=deep)
    assert len(str(target)) > 240
    assert store.load(params, root=deep) is not None


def test_figures_are_written(tmp_path, cheap_run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params, res = cheap_run
    fig, ax = plt.subplots()
    ax.plot(res.t, res.anisotropy)
    ax.set_title('anisotropy decay')
    target = store.save(params, res, figures={'anisotropy decay': fig},
                        root=tmp_path)
    plt.close(fig)
    assert (target / "figures" / "anisotropy-decay.png").is_file()


def test_session_is_separate_from_the_runs(tmp_path):
    target = store.session("s010 comparison", root=tmp_path)
    assert target.parent == tmp_path / "_sessions"
    assert "s010-comparison" in target.name
    assert store.find(root=tmp_path) == []    # sessions are not runs

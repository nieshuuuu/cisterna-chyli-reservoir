import numpy as np
from cc_reservoir.validate.gate import run_gate, summarize

def test_gate_runs_and_reports():
    rows = run_gate(n_per_kind=2, noise_hu=15.0, seed=0)
    assert len(rows) == 2 * 4
    rep = summarize(rows)
    assert "per_kind" in rep and set(rep["per_kind"]) == set(r["kind"] for r in rows)
    for v in rep["per_kind"].values():
        assert set(v) >= {"median_abs_V_err", "median_abs_c_err",
                          "median_abs_m_err", "median_abs_rho", "verdict"}
        assert v["verdict"] in {"V_c_separable", "mass_only"}
    assert rep["verdict"] in {"V_c_separable_all", "mixed"}

def test_thin_family_is_mass_only_but_round_is_separable():
    # The scientific finding: the thin (CC-relevant) morphology is NOT V/c-separable
    # at 15 HU noise, while rounder morphologies are. Locks in the per-kind honesty.
    rep = summarize(run_gate(n_per_kind=10, noise_hu=15.0, seed=0))
    assert rep["per_kind"]["thin"]["verdict"] == "mass_only"
    assert rep["per_kind"]["saccular"]["verdict"] == "V_c_separable"
    assert rep["verdict"] == "mixed"

def test_mass_more_robust_than_volume_on_average():
    rows = run_gate(n_per_kind=6, noise_hu=15.0, seed=1)
    rep = summarize(rows)
    assert rep["median_abs_m_err"] <= rep["median_abs_V_err"] + 0.05

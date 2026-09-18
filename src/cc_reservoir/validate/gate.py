import numpy as np
from ..forward.geometry import DEFAULT_GRID
from ..forward.imaging import forward_patch
from ..forward.morphology import MORPHOLOGIES, sample_cc
from ..estimator.fit import fit_patch
from ..estimator.recovery import vcmass, separability

def run_gate(n_per_kind=15, noise_hu=15.0, grid=DEFAULT_GRID, seed=0):
    """Forward->inverse over the morphology library at realistic noise.
    s_true = 1; the fit knows the base shape and recovers (s, conc)."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in MORPHOLOGIES:
        for _ in range(n_per_kind):
            semi, conc = sample_cc(kind, rng)
            obs = forward_patch(1.0, conc, semi, grid, noise_hu=noise_hu, rng=rng)
            out = fit_patch(obs, semi, grid, init=(1.2, conc * 0.8))
            Vt, _, Mt = vcmass(1.0, conc, semi)
            Ve, _, Me = vcmass(out["s"], out["conc"], semi)
            rows.append(dict(kind=kind, conc=conc,
                             V_err=(Ve - Vt) / Vt,
                             c_err=(out["conc"] - conc) / conc,
                             m_err=(Me - Mt) / Mt,
                             rho_sc=separability(out["cov"])))
    return rows

def summarize(rows):
    """Per-morphology verdict driven by EMPIRICAL recovery error from synthetic
    ground truth. Note: rho_sc (Laplace correlation) is reported but NOT used for
    the verdict -- it stays ~0.5 even when the fit lands in a wrong (mass-preserving)
    basin, so it does not detect the thin-family degeneracy; only V_err/c_err do."""
    kinds = sorted(set(r["kind"] for r in rows))
    per_kind = {}
    for k in kinds:
        rk = [r for r in rows if r["kind"] == k]
        aV = float(np.median([abs(r["V_err"]) for r in rk]))
        ac = float(np.median([abs(r["c_err"]) for r in rk]))
        am = float(np.median([abs(r["m_err"]) for r in rk]))
        rho = float(np.nanmedian([abs(r["rho_sc"]) for r in rk]))
        verdict = "V_c_separable" if (aV < 0.25 and ac < 0.25) else "mass_only"
        per_kind[k] = {"median_abs_V_err": aV, "median_abs_c_err": ac,
                       "median_abs_m_err": am, "median_abs_rho": rho,
                       "verdict": verdict}
    massonly = sorted(k for k, v in per_kind.items() if v["verdict"] == "mass_only")
    separable = sorted(k for k, v in per_kind.items() if v["verdict"] == "V_c_separable")
    return {"per_kind": per_kind,
            "median_abs_V_err": float(np.median([abs(r["V_err"]) for r in rows])),
            "median_abs_c_err": float(np.median([abs(r["c_err"]) for r in rows])),
            "median_abs_m_err": float(np.median([abs(r["m_err"]) for r in rows])),
            "median_abs_rho": float(np.nanmedian([abs(r["rho_sc"]) for r in rows])),
            "separable_kinds": separable, "massonly_kinds": massonly,
            "verdict": "V_c_separable_all" if not massonly else "mixed"}

if __name__ == "__main__":
    rep = summarize(run_gate())
    print("=== CC PV-estimator synthetic-validation gate (per morphology) ===")
    print(f"{'kind':<10} {'|V_err|':>9} {'|c_err|':>9} {'|m_err|':>9} {'|rho|':>7}  verdict")
    for k, v in rep["per_kind"].items():
        print(f"{k:<10} {v['median_abs_V_err']:>9.3f} {v['median_abs_c_err']:>9.3f} "
              f"{v['median_abs_m_err']:>9.3f} {v['median_abs_rho']:>7.3f}  {v['verdict']}")
    print(f"\nseparable (report V and c): {rep['separable_kinds']}")
    print(f"mass_only  (report mass m): {rep['massonly_kinds']}")
    print("OVERALL VERDICT:", rep["verdict"],
          "\n  Phase 2: report V(t) and c(t) only for the separable morphologies;"
          "\n  for mass_only morphologies (thin, CC-relevant), report contrast mass m(t).")

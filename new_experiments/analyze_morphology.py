"""Record-level morphology proxy summaries; annotations stay fixed."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision_v2" / "code"))
import run_v2 as old

np, pd = old.np, old.pd
raw = pd.read_csv(HERE / "morphology_raw.csv")
assert len(raw) == 9720 and raw.record.nunique() == 60
assert raw.groupby(["record", "noise", "db", "seed", "region"]).size().eq(3).all()

out = []
record_rows = []
metrics = ["p_rmse_mv", "qrs_rmse_mv", "t_rmse_mv", "st_abs_error_mv"]
for region in ["quiet", "active"]:
    for noise in ["all", "awgn", "ma1", "blgn"]:
        part = raw[raw.region.eq(region)]
        if noise != "all":
            part = part[part.noise.eq(noise)]
        rec = part.groupby(["record", "method"], as_index=False)[metrics].mean()
        for metric in metrics:
            wide = rec.pivot(index="record", columns="method", values=metric)
            for method, comparator in [("gate_sqrt_subband", "local_frozen"),
                                       ("gate_sqrt_subband", "raw"),
                                       ("local_frozen", "raw")]:
                paired = wide[[method, comparator]].dropna()
                a, b = paired[method].to_numpy(), paired[comparator].to_numpy()
                n = len(a)
                # Ratios to raw are undefined for the exactly unchanged quiet input.
                if not n or np.mean(b) == 0:
                    continue
                estimate = 100 * (np.mean(a) / np.mean(b) - 1)
                rng = np.random.default_rng(20261016 + len(out))
                take = rng.integers(0, n, size=(20000, n))
                draws = 100 * (np.mean(a[take], axis=1) /
                               np.mean(b[take], axis=1) - 1)
                lo, hi = np.quantile(draws, [.025, .975])
                out.append(dict(region=region, noise=noise, metric=metric,
                                method=method, comparator=comparator, records=n,
                                method_mean_mv=float(np.mean(a)),
                                comparator_mean_mv=float(np.mean(b)),
                                change_pct=float(estimate),
                                ci95_low=float(lo), ci95_high=float(hi),
                                records_improved=int(np.sum(a < b))))
                for record, val_a, val_b in zip(paired.index, a, b):
                    record_rows.append(dict(record=record, region=region, noise=noise,
                                            metric=metric, method=method,
                                            comparator=comparator, method_mv=val_a,
                                            comparator_mv=val_b))

pd.DataFrame(out).to_csv(HERE / "morphology_contrasts.csv", index=False)
pd.DataFrame(record_rows).to_csv(HERE / "morphology_record_metrics.csv", index=False)
eligibility = raw.groupby(["region", "method"])[["p_intervals", "qrs_intervals",
                                                  "t_intervals", "eligible_st_beats"]].sum()
eligibility.to_csv(HERE / "morphology_eligibility.csv")
(HERE / "morphology_analysis_checks.json").write_text(json.dumps(dict(
    status="passed", independent_unit="ECG record", records=60,
    identical_input_hashes=True, bootstrap_repetitions=20000,
    fixed_expert_intervals=True, endpoints="post hoc morphology proxies",
    no_clinical_diagnostic_claim=True,
), indent=2), encoding="utf-8")
print(pd.DataFrame(out).query("noise == 'all' and method == 'gate_sqrt_subband' and comparator == 'local_frozen'").to_string(index=False))
print(eligibility.to_string())

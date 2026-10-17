"""Record-level validation summaries without returning to parameter selection."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision_v2" / "code"))
import run_v2 as old
np, pd = old.np, old.pd

data = pd.read_csv(HERE / "validation_raw.csv")
assert len(data) == 3600 and data.record.nunique() == 20
assert data.groupby(["record", "noise", "snr_db", "seed"]).size().eq(15).all()
metrics = ["quiet_nrmse", "active_nrmse", "whole_nmse"]
record = data.groupby(["record", "noise", "method"], as_index=False)[metrics].mean()
record.to_csv(HERE / "validation_record_metrics.csv", index=False)
pooled = data.groupby(["record", "method"], as_index=False)[metrics].mean()
pooled["noise"] = "pooled"
record_all = pd.concat([record, pooled], ignore_index=True)
means = record_all.groupby(["noise", "method"], as_index=False)[metrics].mean()
means.to_csv(HERE / "validation_method_means.csv", index=False)

comparisons = []
for prefix in ["frozen", "new_original", "new_dense"]:
    for comparator in ["local", "gate_linear_subband", "gate_sqrt_global"]:
        comparisons.append((f"{prefix}_gate_sqrt_subband", f"{prefix}_{comparator}"))
comparisons.extend([
    ("new_original_gate_sqrt_subband", "frozen_gate_sqrt_subband"),
    ("new_dense_gate_sqrt_subband", "frozen_gate_sqrt_subband"),
    ("new_dense_gate_sqrt_subband", "new_original_gate_sqrt_subband"),
])
contrasts = []
for noise in ["pooled", "awgn", "ma1"]:
    subset = record_all[record_all.noise == noise]
    for a, b in comparisons:
        for metric in metrics:
            wide = subset.pivot(index="record", columns="method", values=metric)[[a, b]]
            assert wide.shape == (20, 2) and np.isfinite(wide.to_numpy()).all()
            values = wide.to_numpy()
            estimate = float(100 * (values[:, 0].mean() / values[:, 1].mean() - 1))
            rng = np.random.default_rng(20261016 + len(contrasts))
            take = rng.integers(0, 20, size=(20000, 20))
            sample = values[take]
            draws = 100 * (sample[:, :, 0].mean(axis=1) /
                           sample[:, :, 1].mean(axis=1) - 1)
            ci = np.quantile(draws, [.025, .975])
            per_record = 100 * (values[:, 0] / values[:, 1] - 1)
            contrasts.append(dict(noise=noise, target=a, comparator=b, metric=metric,
                                  records=20, change_pct=estimate,
                                  ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                                  record_median_pct=float(np.median(per_record)),
                                  records_above_2pct=int((per_record > 2).sum())
                                  if metric == "active_nrmse" else None,
                                  records_below_20pct_benefit=int((per_record > -20).sum())
                                  if metric == "quiet_nrmse" else None))
pd.DataFrame(contrasts).to_csv(HERE / "validation_contrasts.csv", index=False)

stability = pd.read_csv(HERE / "development_stability.csv")
assert stability.groupby(["grid", "family"]).size().eq(20).all()
frequency = stability.groupby(["grid", "family", "parameters"], as_index=False).size()
frequency.rename(columns={"size": "loo_selection_count"}).to_csv(
    HERE / "development_selection_frequency.csv", index=False)

(HERE / "validation_analysis_checks.json").write_text(json.dumps(dict(
    status="passed", record_unit_count=20, raw_rows=len(data),
    original_grid_and_dense_frozen_before_validation=True,
    equal_record_weight=True, bootstrap_repetitions=20000,
    descriptive_intervals_only=True,
), indent=2), encoding="utf-8")
print(pd.DataFrame(contrasts).query("noise == 'pooled' and metric in ['quiet_nrmse','active_nrmse']")
      [["target", "comparator", "metric", "change_pct", "ci95_low", "ci95_high"]]
      .to_string(index=False))

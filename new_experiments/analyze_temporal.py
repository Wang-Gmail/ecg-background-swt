"""Record-level descriptive analysis of the independent-factor time experiment."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision_v2" / "code"))
import run_v2 as old

np, pd = old.np, old.pd
r = pd.read_csv(HERE / "temporal_raw.csv")
assert len(r) == 4160
assert r.record.nunique() == 20
assert r.groupby(["record", "noise", "seed", "condition"]).size().eq(2).all()
assert (r.quiet_samples >= 1440).all() and (r.active_samples >= 50).all()
assert np.max(np.abs(r.actual_active_snr_db - r.snr_db)) < 1e-9

keys = ["record", "noise", "seed", "condition", "series", "duration_s",
        "center_s", "edge_s", "snr_db", "bursts", "quiet_samples",
        "active_samples", "transition_samples", "positive_fraction",
        "gate_open_active_fraction", "gate_open_quiet_fraction"]
wide = r.pivot(index=keys, columns="method", values=[
    "quiet_nrmse", "active_nrmse", "whole_rmse_mv", "snr_gain_db"
]).reset_index()
wide.columns = ["_".join(str(v) for v in col if v) if isinstance(col, tuple)
                else col for col in wide.columns]
wide["delta_snr_gain_db"] = (
    wide.snr_gain_db_sqrt_subband_frozen - wide.snr_gain_db_local_swt_frozen)
wide["quiet_change_pct"] = 100 * (
    wide.quiet_nrmse_sqrt_subband_frozen / wide.quiet_nrmse_local_swt_frozen - 1)
wide["active_change_pct"] = 100 * (
    wide.active_nrmse_sqrt_subband_frozen / wide.active_nrmse_local_swt_frozen - 1)
wide.to_csv(HERE / "temporal_paired_inputs.csv", index=False)

metrics = ["delta_snr_gain_db", "quiet_change_pct", "active_change_pct",
           "gate_open_active_fraction", "gate_open_quiet_fraction"]
fac = wide[wide.series == "factorial"]
summary = fac.groupby(["noise", "duration_s", "center_s", "edge_s", "snr_db"],
                      as_index=False)[metrics + ["quiet_samples", "active_samples",
                                                "positive_fraction"]].mean()
summary.to_csv(HERE / "temporal_cell_means.csv", index=False)


def interval(values, seed):
    values = np.asarray(values, dtype=float)
    assert len(values) == 20 and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(20000, len(values)))
    return [float(v) for v in np.quantile(values[draws].mean(axis=1), [.025, .975])]


comparisons = [
    ("duration_s", .5, 1.5, "long_minus_short"),
    ("center_s", 3., 7., "late_minus_early"),
    ("edge_s", .025, .15, "wide_minus_narrow"),
    ("snr_db", 0, 6, "6_minus_0_db"),
    ("snr_db", 6, 12, "12_minus_6_db"),
    ("snr_db", 0, 12, "12_minus_0_db"),
]
effects = []
for noise in ["awgn", "ma1"]:
    subset = fac[fac.noise == noise]
    for factor, low, high, label in comparisons:
        for metric in metrics:
            values = subset.groupby(["record", factor])[metric].mean().unstack(factor)
            assert values.shape == (20, 3 if factor == "snr_db" else 2)
            diff = values[high] - values[low]
            ci = interval(diff, 20261016 + len(effects))
            effects.append(dict(noise=noise, factor=factor, contrast=label,
                                metric=metric, records=20,
                                mean_high_minus_low=float(diff.mean()),
                                ci95_low=ci[0], ci95_high=ci[1]))
pd.DataFrame(effects).to_csv(HERE / "temporal_factor_effects.csv", index=False)

frag = wide[wide.series == "fragmentation"]
for column in ["quiet_samples", "active_samples", "positive_fraction"]:
    spread = frag.groupby(["record", "noise", "seed"])[column].agg(lambda z: z.max() - z.min())
    assert np.max(np.abs(spread)) < 1e-12, column
fragment_effects = []
for noise in ["awgn", "ma1"]:
    subset = frag[frag.noise == noise]
    for metric in metrics:
        values = subset.groupby(["record", "bursts"])[metric].mean().unstack("bursts")
        diff = values[4] - values[1]
        ci = interval(diff, 20261116 + len(fragment_effects))
        fragment_effects.append(dict(noise=noise, metric=metric, records=20,
                                     mean_four_minus_one=float(diff.mean()),
                                     ci95_low=ci[0], ci95_high=ci[1]))
pd.DataFrame(fragment_effects).to_csv(HERE / "temporal_fragmentation_effects.csv", index=False)

json_path = HERE / "temporal_analysis_checks.json"
json_path.write_text(json.dumps(dict(status="passed", records=20, input_cases=len(wide),
    paired_factorial_cases=len(fac), paired_fragmentation_cases=len(frag),
    actual_snr_max_abs_error_db=float(np.max(np.abs(r.actual_active_snr_db-r.snr_db))),
    quiet_min_samples=int(r.quiet_samples.min()), active_min_samples=int(r.active_samples.min()),
    fragmentation_equal_samples=True, bootstrap_repetitions=20000,
    independent_unit="ECG record", inference="descriptive, local follow-up"),
    indent=2), encoding="utf-8")
print(pd.DataFrame(fragment_effects).to_string(index=False))

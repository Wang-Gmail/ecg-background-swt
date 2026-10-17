"""Subject-level, paired descriptive analysis of the SimEMG follow-up."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision_v2" / "code"))
import run_v2 as old
np, pd = old.np, old.pd

raw = pd.read_csv(HERE / "simemg_raw.csv")
assert len(raw) == 660 and raw.patient.nunique() == 14
assert raw.groupby("file").size().eq(6).all()
assert raw.groupby(["patient", "recording", "electrode"]).size().eq(6).all()
electrode_subject = raw.groupby(["patient", "electrode", "method"],
                                as_index=False)[["nrmse", "snr_gain_db"]].mean()
electrode_subject.to_csv(HERE / "simemg_subject_electrode.csv", index=False)
overall = electrode_subject.groupby(["patient", "method"],
                                    as_index=False)[["nrmse", "snr_gain_db"]].mean()
overall["electrode"] = "all_equal_electrode"
joined = pd.concat([electrode_subject, overall], ignore_index=True)
joined.groupby(["electrode", "method"], as_index=False)[["nrmse", "snr_gain_db"]].mean().to_csv(
    HERE / "simemg_method_means.csv", index=False)

comparisons = [("gate_sqrt_subband", m) for m in
               ["raw", "local_frozen", "gate_linear_subband", "gate_sqrt_global"]]
comparisons.extend((m, "raw") for m in
                   ["local_frozen", "gate_linear_subband", "gate_sqrt_global", "gate_linear_global"])
rows = []
for electrode in ["all_equal_electrode", "Ag-AgCl", "ORB", "lead I"]:
    group = joined[joined.electrode == electrode]
    for method, comparator in comparisons:
        table = group.pivot(index="patient", columns="method", values="nrmse")[[method, comparator]].dropna()
        values = table.to_numpy()
        patients = len(values)
        assert patients >= 13
        estimate = float(100 * (values[:, 0].mean() / values[:, 1].mean() - 1))
        rng = np.random.default_rng(20261016 + len(rows))
        take = rng.integers(0, patients, size=(20000, patients))
        sampled = values[take]
        draws = 100 * (sampled[:, :, 0].mean(axis=1) /
                       sampled[:, :, 1].mean(axis=1) - 1)
        ci = np.quantile(draws, [.025, .975])
        rows.append(dict(electrode=electrode, method=method, comparator=comparator,
                         subjects=patients, change_pct=estimate,
                         ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                         subjects_improved=int((values[:, 0] < values[:, 1]).sum())))
pd.DataFrame(rows).to_csv(HERE / "simemg_contrasts.csv", index=False)
(HERE / "simemg_analysis_checks.json").write_text(json.dumps(dict(
    status="passed", files=110, subjects=14, recordings=37,
    nested_unit="subject", repeated_recordings_and_electrodes_not_independent=True,
    bootstrap_repetitions=20000, intervals="descriptive",
    no_quiet_active_claim=True, no_clinical_truth_claim=True,
), indent=2), encoding="utf-8")
print(pd.DataFrame(rows).query("electrode == 'all_equal_electrode'").to_string(index=False))

"""Record-level paired summaries of the independent physical noise tests."""
from __future__ import annotations

import json
from pathlib import Path
import run_extension as ext

HERE = Path(__file__).resolve().parent
np, pd = ext.np, ext.pd
manifest = json.loads((HERE / 'independent_noise_manifest.json').read_text(encoding='utf-8'))
official_sha256 = {
    'em.dat': '4f9416a04b2bed98081ee3118c1282f9e35321bfd5da96ce48056a3fe147358d',
    'em.hea': '0bdfa989de0dd06486159f2c36bd80abd684d951c3930daea2ee46572fd20d49',
    'bw.dat': '15f68280aba8b006f5497c14da44b558e68caaabc2afc5c4b710eea779b7512c',
    'bw.hea': 'e3ea3b09fb8d65877cf6ae4dec213b9693243c435333a150c14f6b551f7b7d61',
}
assert manifest['noise_source_sha256'] == official_sha256
raw = pd.read_csv(HERE / 'independent_noise_raw.csv')
assert len(raw) == 960 and raw.record.nunique() == 20
assert raw.groupby(['record', 'noise', 'snr_db', 'seed']).input_sha256.nunique().eq(1).all()
assert raw.groupby(['record', 'noise', 'snr_db', 'seed']).size().eq(4).all()
assert np.max(np.abs(raw.actual_active_snr_db - raw.snr_db)) < 1e-9

metrics = ['quiet_nrmse', 'active_nrmse', 'whole_rmse_mv', 'snr_gain_db']
rows = []
for noise in ['em', 'bw']:
    means = raw.loc[raw.noise == noise].groupby(['record', 'method'], as_index=False)[metrics].mean()
    for comparator in ['local_swt_frozen', 'linear_subband_frozen', 'raw']:
        for metric in metrics:
            # The injected-noise-free quiet input is exactly the reference.
            # A percentage relative to its zero error is undefined.
            if comparator == 'raw' and metric == 'quiet_nrmse':
                continue
            wide = means.pivot(index='record', columns='method', values=metric)
            paired = wide[['sqrt_subband_frozen', comparator]].dropna()
            a, b = paired.to_numpy().T
            assert len(a) == 20
            if metric == 'snr_gain_db':
                estimate = float(np.mean(a - b))
                unit = 'dB difference'
            else:
                estimate = float(100 * (np.mean(a) / np.mean(b) - 1))
                unit = 'percent change'
            rng = np.random.default_rng(20261016 + len(rows))
            take = rng.integers(0, len(a), size=(20000, len(a)))
            if metric == 'snr_gain_db':
                draws = np.mean(a[take] - b[take], axis=1)
            else:
                draws = 100 * (np.mean(a[take], axis=1) / np.mean(b[take], axis=1) - 1)
            lo, hi = np.quantile(draws, [.025, .975])
            rows.append(dict(noise=noise, method='sqrt_subband_frozen', comparator=comparator,
                             metric=metric, unit=unit, records=len(a), method_mean=float(np.mean(a)),
                             comparator_mean=float(np.mean(b)), estimate=estimate,
                             ci95_low=float(lo), ci95_high=float(hi),
                             records_better=int(np.sum(a > b if metric == 'snr_gain_db' else a < b))))

summary = pd.DataFrame(rows)
summary.to_csv(HERE / 'independent_noise_contrasts.csv', index=False)
(HERE / 'independent_noise_analysis_checks.json').write_text(json.dumps(dict(
    status='passed', paired_inputs=True, record_unit=True, records=20,
    bootstrap_repetitions=20000, intervals='descriptive', target_snr_asserted=True,
    official_physionet_sha256_matched=True,
    no_new_muscle_source_claim=True, no_natural_contamination_claim=True), indent=2), encoding='utf-8')
print(summary.query("comparator == 'local_swt_frozen' and metric in ['quiet_nrmse', 'active_nrmse', 'whole_rmse_mv']").to_string(index=False))

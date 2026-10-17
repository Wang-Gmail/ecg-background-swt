"""Prespecified EM and BW physical-recording tests with frozen methods."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import run_extension as ext

ROOT, HERE = ext.ROOT, ext.HERE
np, pd, wfdb, old = ext.np, ext.pd, ext.wfdb, ext.old
FS = ext.FS


def main() -> None:
    records = sorted(ext.allocation().query("group == 'validation'").record.tolist())
    assert len(records) == 20
    env = ext.envelope([(2., 1.3), (6., 1.3)], .1)
    valid = np.zeros(3600, dtype=bool)
    valid[FS:-FS] = True
    quiet, active = valid & (env == 0), valid & (env > .999)
    assert quiet.sum() > 4 * FS and active.sum() > 100
    source_hashes = {}
    rows = []
    for kind in ['em', 'bw']:
        path = ROOT / 'data' / 'nstdb' / kind
        source_hashes[f'{kind}.dat'] = ext.sha(path.with_suffix('.dat'))
        source_hashes[f'{kind}.hea'] = ext.sha(path.with_suffix('.hea'))
        source = wfdb.rdrecord(str(path), channels=[1])
        assert source.fs == FS and len(source.p_signal) >= 1780 * FS
        pool = source.p_signal[:, 0]
        for record, snr_db, seed in itertools.product(records, [0, 6, 12], [9401, 9402]):
            x = ext.clip(record)
            rng = np.random.default_rng(seed * 10007 + record * 101)
            offset = int(rng.integers(1080 * FS, 1780 * FS - len(x)))
            z = pool[offset:offset + len(x)].copy()
            z -= z.mean()
            noise_power = np.mean((z[active] * env[active]) ** 2)
            assert noise_power > 1e-18
            scale = np.sqrt(np.mean(x[active] ** 2) / (10 ** (snr_db / 10) * noise_power))
            y = x + scale * z * env
            actual = float(10 * np.log10(np.mean(x[active] ** 2) / np.mean((y[active] - x[active]) ** 2)))
            assert abs(actual - snr_db) < 1e-9
            features = old.ecg.features(y)
            outputs = {
                'raw': y,
                'sqrt_subband_frozen': old.gated(y, features, **ext.FROZEN['chosen']['gate_sqrt_subband']),
                'linear_subband_frozen': old.gated(y, features, **ext.FROZEN['chosen']['gate_linear_subband']),
                'local_swt_frozen': old.ecg.apply(y, features, dict(family='local', alpha=3, shrink='firm')),
            }
            key = dict(record=record, noise=kind, snr_db=snr_db, seed=seed,
                       offset_sample=offset, quiet_samples=int(quiet.sum()), active_samples=int(active.sum()),
                       actual_active_snr_db=actual,
                       input_sha256=hashlib.sha256(y.tobytes()).hexdigest())
            for method, output in outputs.items():
                metrics = ext.regional_metrics(x, y, output, valid, quiet, active)
                assert all(np.isfinite(v) for v in metrics.values())
                rows.append(dict(**key, method=method, **metrics))
    frame = pd.DataFrame(rows)
    assert len(frame) == 960 and frame.groupby(['record', 'noise', 'snr_db', 'seed']).size().eq(4).all()
    frame.to_csv(HERE / 'independent_noise_raw.csv', index=False)
    ext.write_json(HERE / 'independent_noise_manifest.json', dict(
        plan='new_experiments/PLAN.md: E4 EM/BW addendum', status='complete',
        records=records, inputs=240, rows=960, fs=FS,
        noise_source_sha256=source_hashes, frozen_config_sha256=ext.sha(ext.FROZEN_PATH),
        code_sha256=ext.sha(Path(__file__)), no_reselection=True,
        source_recording_different=True, noise_type_different=True))
    print('Wrote 240 inputs and 960 method rows')


if __name__ == '__main__':
    main()

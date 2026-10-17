"""Frozen-method validation on simultaneous real EMG-contaminated ECG pairs."""
from pathlib import Path
import hashlib
import io
import itertools
import json
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision_v2" / "code"))
import run_v2 as old
from scipy.io import loadmat

np, pd, signal = old.np, old.pd, old.signal
archive = HERE / "public_data" / "simemg-1.zip"
frozen_path = ROOT / "revision_v2" / "results" / "frozen_config.json"
frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["chosen"]
pattern = re.compile(r"P(\d+)_(\d+)_(Ag-AgCl|lead I|ORB)\.mat$")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(x, y, z):
    valid = slice(360, -360)
    ref = x[valid]
    raw_err = y[valid] - ref
    out_err = z[valid] - ref
    reference_rms = float(np.sqrt(np.mean(ref ** 2)))
    raw_rmse = float(np.sqrt(np.mean(raw_err ** 2)))
    out_rmse = float(np.sqrt(np.mean(out_err ** 2)))
    return dict(reference_rms_mv=reference_rms, raw_rmse_mv=raw_rmse,
                rmse_mv=out_rmse, nrmse=out_rmse / reference_rms,
                snr_gain_db=float(20 * np.log10(raw_rmse / max(out_rmse, 1e-30))))


with zipfile.ZipFile(archive) as zip_file:
    assert zip_file.testzip() is None
    files = [x for x in zip_file.namelist() if x.lower().endswith(".mat")]
    assert len(files) == 110
    parsed = []
    references = {}
    rows = []
    for name in sorted(files):
        match = pattern.search(name)
        assert match, name
        patient, recording, electrode = int(match[1]), int(match[2]), match[3]
        content = zip_file.read(name)
        data = loadmat(io.BytesIO(content))
        variables = [value for key, value in data.items() if not key.startswith("__")]
        assert len(variables) == 1
        matrix = np.asarray(variables[0], dtype=float)
        assert matrix.shape == (15000, 2) and np.isfinite(matrix).all()
        group = (patient, recording)
        if group in references:
            assert np.array_equal(references[group], matrix[:, 0])
        else:
            references[group] = matrix[:, 0].copy()
        # The dataset records digital amplitude at 200 units per mV.
        x = signal.resample_poly(matrix[:, 0] / 200., 18, 25)
        y = signal.resample_poly(matrix[:, 1] / 200., 18, 25)
        x -= x.mean()
        y -= y.mean()
        assert len(x) == len(y) == 10800
        features = old.ecg.features(y)
        outputs = {"raw": y, "local_frozen": old.ecg.apply(
            y, features, dict(family="local", alpha=3, shrink="firm"))}
        for family in old.FAMILIES:
            outputs[family] = old.gated(y, features, **frozen[family])
        for method, z in outputs.items():
            rows.append(dict(patient=patient, recording=recording,
                             electrode=electrode, file=Path(name).name,
                             file_sha256=hashlib.sha256(content).hexdigest(),
                             method=method, **metric(x, y, z)))
        parsed.append((patient, recording, electrode))
        print("simemg", Path(name).name, flush=True)

assert len(set(p for p, _, _ in parsed)) == 14
assert len(references) == 37
assert len(rows) == 110 * 6
raw = pd.DataFrame(rows)
raw.to_csv(HERE / "simemg_raw.csv", index=False)
metadata = dict(status="completed", source="SimEMG V1",
                doi="10.17632/yx5pb66hwz.1", subjects=14,
                simultaneous_recording_groups=37, paired_files=110,
                methods=sorted(raw.method.unique().tolist()),
                rows=len(raw), shared_reference_column_exact=True,
                sample_rate_original_hz=500, sample_rate_analysis_hz=360,
                amplitude_units_per_mv=200, valid_margin_s=1,
                archive_sha256=sha(archive), frozen_config_sha256=sha(frozen_path),
                code_sha256=sha(Path(__file__)), python=sys.version,
                reference_caution="Shoulder electrode reference, not physiological noise-free truth")
(HERE / "simemg_metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

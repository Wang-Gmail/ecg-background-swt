"""Fixed-annotation morphology proxies on the frozen LUDB holdout."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision_v2" / "code"))
import run_v2 as old
np, pd, wfdb = old.np, old.pd, old.wfdb

frozen_path = ROOT / "revision_v2" / "results" / "frozen_config.json"
frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["chosen"]
saved_inputs = pd.read_csv(ROOT / "revision_v2" / "results" / "holdout_inputs.csv")
assert len(saved_inputs) == 1620
saved_hashes = {(int(row.record), row.noise, int(row.db), int(row.seed)): row.input_sha256
                for row in saved_inputs.itertuples()}
assert len(saved_hashes) == 1620


def intervals_for(record: str):
    ann = wfdb.rdann(str(ROOT / "revision_v2" / "data" / "ludb_holdout" / record), "ii")
    events = []
    begin = None
    label = None
    for sample, symbol in zip(ann.sample, ann.symbol):
        index = int(round(int(sample) * 18 / 25))
        if symbol == "(":
            begin, label = index, None
        elif symbol in ["p", "N", "t"] and begin is not None:
            label = {"p": "P", "N": "QRS", "t": "T"}[symbol]
        elif symbol == ")" and begin is not None and label is not None:
            end = index + 1
            if 0 <= begin < end <= 3600:
                events.append((begin, end, label))
            begin, label = None, None
    waves = {name: [(a, b) for a, b, label in events if label == name]
             for name in ["P", "QRS", "T"]}
    qrs = waves["QRS"]
    beats = []
    for i, (qstart, qend) in enumerate(qrs):
        previous_qrs_end = qrs[i-1][1] if i else 0
        next_qrs_start = qrs[i+1][0] if i+1 < len(qrs) else 3600
        preceding_p = [pair for pair in waves["P"]
                       if previous_qrs_end <= pair[1] < qstart]
        following_t = [pair for pair in waves["T"]
                       if qend < pair[0] < next_qrs_start]
        if not preceding_p or not following_t:
            continue
        p_end = preceding_p[-1][1]
        t_start = following_t[0][0]
        st_start = qend + round(.06 * 360)
        st_end = qend + round(.08 * 360) + 1
        if qstart - p_end < 10 or qstart - p_end > round(.4 * 360):
            continue
        if st_end > t_start or st_start >= st_end:
            continue
        beats.append((p_end, qstart, st_start, st_end))
    return waves, beats


def wave_mask(intervals, region):
    mask = np.zeros(3600, dtype=bool)
    count = 0
    for start, end in intervals:
        if region[start:end].all():
            mask[start:end] = True
            count += 1
    return mask, count


def st_difference(x, z, beat_spans, region):
    errors = []
    for p_end, q_start, st_start, st_end in beat_spans:
        if not (region[p_end:q_start].all() and region[st_start:st_end].all()):
            continue
        x_st = float(np.mean(x[st_start:st_end]) - np.median(x[p_end:q_start]))
        z_st = float(np.mean(z[st_start:st_end]) - np.median(z[p_end:q_start]))
        errors.append(z_st - x_st)
    return errors


old.noise_inputs.cache = old.noise_inputs()
rows = []
for clip in old.clips():
    record = int(clip["record"])
    waves, beats = intervals_for(clip["record"])
    x = clip["x"]
    for noise in ["awgn", "ma1", "blgn"]:
        for db in [0, 6, 12]:
            for seed in [6101, 6102, 6103]:
                y, masks, info = old.corrupt(clip, noise, db, seed)
                assert info["input_sha256"] == saved_hashes[(record, noise, db, seed)]
                features = old.ecg.features(y)
                outputs = {
                    "raw": y,
                    "local_frozen": old.ecg.apply(
                        y, features, dict(family="local", alpha=3, shrink="firm")),
                    "gate_sqrt_subband": old.gated(
                        y, features, **frozen["gate_sqrt_subband"]),
                }
                for region_name in ["quiet", "active"]:
                    region = masks[region_name] & masks["valid"]
                    wm = {key: wave_mask(value, region)
                          for key, value in waves.items()}
                    for method, z in outputs.items():
                        error = (z-x) ** 2
                        item = dict(record=record, noise=noise, db=db, seed=seed,
                                    region=region_name, method=method,
                                    eligible_st_beats=0)
                        for wave, (mask, count) in wm.items():
                            item[f"{wave.lower()}_intervals"] = count
                            item[f"{wave.lower()}_samples"] = int(mask.sum())
                            item[f"{wave.lower()}_rmse_mv"] = (
                                float(np.sqrt(np.mean(error[mask]))) if count else None)
                        st_errors = st_difference(x, z, beats, region)
                        item["eligible_st_beats"] = len(st_errors)
                        item["st_abs_error_mv"] = (
                            float(np.mean(np.abs(st_errors))) if st_errors else None)
                        item["st_signed_error_mv"] = (
                            float(np.mean(st_errors)) if st_errors else None)
                        rows.append(item)
    print("morphology record", record, "rows", len(rows), flush=True)

assert len(rows) == 60 * 27 * 2 * 3
pd.DataFrame(rows).to_csv(HERE / "morphology_raw.csv", index=False)
metadata = dict(status="completed", records=60, input_cases=1620,
                rows=len(rows), all_input_hashes_match=True,
                method_parameters="original frozen", analysis="post hoc morphology proxies",
                st_offset_ms=[60, 80], pr_baseline="expert P offset to QRS onset",
                clinical_diagnostic_validity=False,
                frozen_config_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                python=sys.version)
(HERE / "morphology_metadata.json").write_text(json.dumps(
    metadata, indent=2, ensure_ascii=False), encoding="utf-8")

"""Reproducible, locally planned follow-up experiments for the IPOR paper."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision_v2" / "code"))
import run_v2 as old  # Loads the project's locked Python 3.12 scientific dependencies.

np, pd, signal, wfdb = old.np, old.pd, old.signal, old.wfdb
FROZEN_PATH = ROOT / "revision_v2" / "results" / "frozen_config.json"
FROZEN = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
FS = 360


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def allocation() -> pd.DataFrame:
    path = HERE / "record_allocation.csv"
    if path.exists():
        table = pd.read_csv(path)
    else:
        ids = np.arange(3, 181, 3)
        assert len(ids) == 60
        rng = np.random.default_rng(20261016)
        shuffled = rng.permutation(ids)
        table = pd.DataFrame(
            [(int(record), group) for group, chunk in zip(
                ["development", "validation", "temporal"],
                np.split(shuffled, 3),
            ) for record in chunk], columns=["record", "group"]
        ).sort_values("record")
        table.to_csv(path, index=False)
    assert set(table.record) == set(range(3, 181, 3))
    assert table.groupby("group").size().to_dict() == {
        "development": 20, "temporal": 20, "validation": 20,
    }
    assert not set(table.record) & set(range(1, 179, 3))
    assert not set(table.record) & set(range(2, 180, 3))
    for rec in table.record:
        for ext in ["dat", "hea"]:
            assert (ROOT / "data" / "ludb" / f"{rec}.{ext}").is_file()
    return table


def clip(record: int) -> np.ndarray:
    path = ROOT / "data" / "ludb" / str(record)
    source = wfdb.rdrecord(str(path), channels=[1])
    assert source.fs == 500 and source.sig_name[0].lower() == "ii"
    x = signal.resample_poly(source.p_signal[:, 0], 18, 25)
    x -= x.mean()
    assert len(x) == 3600 and np.isfinite(x).all()
    return x


def envelope(bursts: list[tuple[float, float]], edge_s: float) -> np.ndarray:
    env = np.zeros(10 * FS)
    for center, duration in bursts:
        lo = round((center - duration / 2) * FS)
        hi = round((center + duration / 2) * FS)
        assert FS < lo < hi < 9 * FS
        env[lo:hi] = 1
        ramp = round(edge_s * FS)
        assert 0 <= 2 * ramp < hi - lo
        if ramp:
            edge = .5 - .5 * np.cos(np.linspace(0, np.pi, ramp))
            env[lo:lo + ramp] = edge
            env[hi - ramp:hi] = edge[::-1]
    return env


def temporal_conditions() -> list[dict]:
    conditions = [
        dict(series="factorial", duration_s=d, center_s=c, edge_s=e,
             snr_db=snr, bursts=[(c, d)])
        for d, c, e, snr in itertools.product(
            [.5, 1.5], [3., 7.], [.025, .15], [0, 6, 12]
        )
    ]
    conditions.extend([
        dict(series="fragmentation", duration_s=2., center_s=5., edge_s=0.,
             snr_db=6, bursts=[(5., 2.)]),
        dict(series="fragmentation", duration_s=.5, center_s=None, edge_s=0.,
             snr_db=6, bursts=[(2., .5), (4., .5), (6., .5), (8., .5)]),
    ])
    assert len(conditions) == 26
    return conditions


def base_noise(record: int, kind: str, seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 10007 + record * 101)
    if kind == "awgn":
        return rng.normal(size=n)
    assert kind == "ma1"
    pool = old.noise_inputs.cache["ma1"]
    lo, hi = 1080 * FS, 1780 * FS - n
    offset = int(rng.integers(lo, hi))
    z = pool[offset:offset + n].copy()
    z -= z.mean()
    return z


def regional_metrics(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     valid: np.ndarray, quiet: np.ndarray,
                     active: np.ndarray) -> dict:
    reference_rms = float(np.sqrt(np.mean(x[valid] ** 2)))
    error = z - x
    q = float(np.sqrt(np.mean(error[quiet] ** 2)))
    a = float(np.sqrt(np.mean(error[active] ** 2)))
    whole = float(np.sqrt(np.mean(error[valid] ** 2)))
    input_mse = float(np.mean((y[valid] - x[valid]) ** 2))
    return dict(quiet_nrmse=q / reference_rms, active_nrmse=a / reference_rms,
                quiet_rmse_mv=q, active_rmse_mv=a, whole_rmse_mv=whole,
                snr_gain_db=float(10 * np.log10(input_mse / max(whole ** 2, 1e-30))))


def run_temporal() -> None:
    table = allocation()
    records = sorted(table.loc[table.group == "temporal", "record"].tolist())
    old.noise_inputs.cache = old.noise_inputs()
    conditions = temporal_conditions()
    chosen = FROZEN["chosen"]["gate_sqrt_subband"]
    rows = []
    valid = np.zeros(3600, dtype=bool)
    valid[FS:-FS] = True
    for rec in records:
        x = clip(rec)
        for kind, seed in itertools.product(["awgn", "ma1"], [8101, 8102]):
            raw_noise = base_noise(rec, kind, seed, len(x))
            for i, condition in enumerate(conditions):
                env = envelope(condition["bursts"], condition["edge_s"])
                quiet = valid & (env == 0)
                active = valid & (env > .999)
                transition = valid & (env > 0) & ~active
                assert quiet.sum() >= 4 * FS and active.sum() >= 50
                scale = np.sqrt(np.mean(x[active] ** 2) /
                                (10 ** (condition["snr_db"] / 10) *
                                 np.mean((raw_noise[active] * env[active]) ** 2)))
                y = x + scale * raw_noise * env
                actual_snr = float(10 * np.log10(
                    np.mean(x[active] ** 2) /
                    np.mean((y[active] - x[active]) ** 2)))
                assert abs(actual_snr - condition["snr_db"]) < 1e-9
                features = old.ecg.features(y)
                sq = features["local"][-1][features["pad"]:features["pad"] + len(x)]
                gate_open = sq > chosen["c"] * features["background"][-1]
                outputs = {
                    "sqrt_subband_frozen": old.gated(y, features, **chosen),
                    "local_swt_frozen": old.ecg.apply(
                        y, features, dict(family="local", alpha=3, shrink="firm")),
                }
                keys = dict(record=rec, noise=kind, seed=seed, condition=i,
                            series=condition["series"], duration_s=condition["duration_s"],
                            center_s=condition["center_s"], edge_s=condition["edge_s"],
                            snr_db=condition["snr_db"], bursts=len(condition["bursts"]),
                            quiet_samples=int(quiet.sum()), active_samples=int(active.sum()),
                            transition_samples=int(transition.sum()),
                            positive_fraction=float(np.mean(env[valid] > 0)),
                            actual_active_snr_db=actual_snr,
                            gate_open_active_fraction=float(np.mean(gate_open[active])),
                            gate_open_quiet_fraction=float(np.mean(gate_open[quiet])),
                            input_sha256=hashlib.sha256(y.tobytes()).hexdigest())
                for name, output in outputs.items():
                    rows.append(dict(**keys, method=name,
                                     **regional_metrics(x, y, output, valid, quiet, active)))
        print("temporal record", rec, "rows", len(rows), flush=True)
    expected = 20 * 2 * 2 * 26 * 2
    assert len(rows) == expected
    data = pd.DataFrame(rows)
    data.to_csv(HERE / "temporal_raw.csv", index=False)
    write_json(HERE / "temporal_metadata.json", dict(
        status="completed", records=records, unique_inputs=expected // 2,
        rows=expected, conditions=conditions,
        allocation_sha256=sha(HERE / "record_allocation.csv"),
        frozen_config_sha256=sha(FROZEN_PATH),
        code_sha256=sha(Path(__file__)), python=sys.version,
        numpy=np.__version__, scipy=old.rr.scipy.__version__,
    ))


def development_candidates() -> list[dict]:
    alphas = [1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6.]
    cs = [round(.5 + .25 * i, 2) for i in range(11)]
    original_a = {1., 2., 3., 4., 6.}
    original_c = {.5, 1., 1.5, 2., 2.5, 3.}
    candidates = []
    for family, base in old.FAMILIES.items():
        for a, c in itertools.product(alphas, cs):
            candidates.append(dict(candidate=len(candidates), family=family,
                                   parameters=dict(**base, alpha=a, c=c),
                                   original_grid=a in original_a and c in original_c))
    for a in alphas:
        candidates.append(dict(candidate=len(candidates), family="local",
                               parameters=dict(alpha=a), original_grid=a in original_a))
    assert len(candidates) == 405
    assert sum(c["original_grid"] for c in candidates) == 125
    return candidates


def score_case(x: np.ndarray, y: np.ndarray, masks: dict, z: np.ndarray) -> dict:
    valid, quiet, active = (masks[k] for k in ["valid", "quiet", "active"])
    power = np.mean(x[valid] ** 2)
    err = (z - x) ** 2
    return dict(active_nrmse=float(np.sqrt(np.mean(err[active]) / power)),
                quiet_nrmse=float(np.sqrt(np.mean(err[quiet]) / power)),
                whole_nmse=float(np.mean(err[valid]) / power))


def select_parameters(scores: pd.DataFrame, candidates: list[dict],
                      original_only: bool) -> dict:
    catalog = pd.DataFrame([
        dict(candidate=c["candidate"], family=c["family"],
             original_grid=c["original_grid"]) for c in candidates
    ])
    joined = scores.merge(catalog, on="candidate", validate="many_to_one")
    if original_only:
        joined = joined[joined.original_grid]
    metrics = ["active_nrmse", "quiet_nrmse", "whole_nmse"]
    per_record = joined.groupby(["record", "candidate", "family"],
                                as_index=False)[metrics].mean()
    means = per_record.groupby(["candidate", "family"],
                               as_index=False)[metrics].mean()
    local = means[means.family == "local"]
    cap = 1.01 * float(local.active_nrmse.min())
    selected = []
    for family in [*old.FAMILIES, "local"]:
        group = means[means.family == family].copy()
        if family == "local":
            row = group.sort_values(["active_nrmse", "candidate"]).iloc[0]
            feasible_count = len(group)
        else:
            feasible = group[group.active_nrmse <= cap]
            feasible_count = len(feasible)
            row = (feasible.sort_values(["quiet_nrmse", "active_nrmse", "candidate"])
                   if feasible_count else group.sort_values(["whole_nmse", "candidate"])).iloc[0]
        identity = candidates[int(row.candidate)]
        selected.append(dict(family=family, candidate=int(row.candidate),
                             parameters=identity["parameters"], candidates=len(group),
                             feasible_candidates=feasible_count,
                             active_nrmse=float(row.active_nrmse),
                             quiet_nrmse=float(row.quiet_nrmse),
                             whole_nmse=float(row.whole_nmse)))
    return dict(active_cap=cap, selected=selected)


def run_development() -> None:
    records = sorted(allocation().query("group == 'development'").record.tolist())
    candidates = development_candidates()
    write_json(HERE / "development_candidates.json", candidates)
    old.noise_inputs.cache = old.noise_inputs()
    rows = []
    for rec in records:
        x = clip(rec)
        info = dict(record=str(rec), x=x)
        for kind, snr, seed in itertools.product(["awgn", "ma1"], [0, 6, 12], [9101, 9102]):
            y, masks, meta = old.corrupt(info, kind, snr, seed)
            features = old.ecg.features(y)
            for candidate in candidates:
                p = candidate["parameters"]
                if candidate["family"] == "local":
                    z = old.ecg.apply(y, features, dict(family="local", alpha=p["alpha"],
                                                        shrink="firm"))
                else:
                    z = old.gated(y, features, **p)
                rows.append(dict(record=rec, noise=kind, snr_db=snr, seed=seed,
                                 input_sha256=meta["input_sha256"],
                                 candidate=candidate["candidate"],
                                 **score_case(x, y, masks, z)))
        print("development record", rec, "rows", len(rows), flush=True)
    assert len(rows) == 20 * 2 * 3 * 2 * len(candidates)
    data = pd.DataFrame(rows)
    data.to_csv(HERE / "development_raw.csv", index=False)
    original = select_parameters(data, candidates, True)
    dense = select_parameters(data, candidates, False)
    write_json(HERE / "development_selection.json", dict(original_grid=original,
                                                         dense_grid=dense))
    # Leave-one-record-out stability retains the record as the independent unit.
    stability = []
    for rec in records:
        train = data[data.record != rec]
        for grid, original_only in [("original", True), ("dense", False)]:
            selection = select_parameters(train, candidates, original_only)
            for item in selection["selected"]:
                stability.append(dict(left_out_record=rec, grid=grid,
                                      family=item["family"], candidate=item["candidate"],
                                      parameters=json.dumps(item["parameters"], sort_keys=True),
                                      feasible_candidates=item["feasible_candidates"]))
    pd.DataFrame(stability).to_csv(HERE / "development_stability.csv", index=False)
    official = {}
    for line in (ROOT / "data" / "ludb" / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            official[parts[1]] = parts[0]
    for rec in records:
        for ext in ["dat", "hea"]:
            key = f"data/{rec}.{ext}"
            assert sha(ROOT / "data" / "ludb" / f"{rec}.{ext}") == official[key]
    write_json(HERE / "development_metadata.json", dict(
        status="completed", records=records, inputs=20 * 12,
        candidates=len(candidates), rows=len(rows),
        official_file_hashes_verified=len(records) * 2,
        allocation_sha256=sha(HERE / "record_allocation.csv"),
        frozen_config_sha256=sha(FROZEN_PATH), code_sha256=sha(Path(__file__)),
        python=sys.version, numpy=np.__version__, scipy=old.rr.scipy.__version__,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["allocate", "temporal", "development"])
    args = parser.parse_args()
    if args.phase == "allocate":
        print(allocation().groupby("group").record.agg(["count", "min", "max"]))
    elif args.phase == "temporal":
        run_temporal()
    else:
        run_development()

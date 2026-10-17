"""One-pass evaluation on the untouched third-modulo-three LUDB validation split."""
from pathlib import Path
import hashlib
import itertools
import json
import sys

HERE = Path(__file__).resolve().parent
from run_extension import ROOT, FROZEN, FROZEN_PATH, allocation, clip, old, score_case, sha, write_json

np, pd = old.np, old.pd
selection_path = HERE / "development_selection.json"
selection = json.loads(selection_path.read_text(encoding="utf-8"))
records = sorted(allocation().query("group == 'validation'").record.tolist())
assert len(records) == 20
methods = {}
for family, parameters in FROZEN["chosen"].items():
    if family.startswith("gate_"):
        methods[f"frozen_{family}"] = (family, parameters)
methods["frozen_local"] = ("local", {"alpha": 3.})
for label, key in [("new_original", "original_grid"), ("new_dense", "dense_grid")]:
    for item in selection[key]["selected"]:
        methods[f"{label}_{item['family']}"] = (item["family"], item["parameters"])
assert len(methods) == 15

old.noise_inputs.cache = old.noise_inputs()
rows = []
for rec in records:
    x = clip(rec)
    info = dict(record=str(rec), x=x)
    for kind, snr, seed in itertools.product(["awgn", "ma1"], [0, 6, 12], [9201, 9202]):
        y, masks, meta = old.corrupt(info, kind, snr, seed)
        features = old.ecg.features(y)
        for name, (family, parameters) in methods.items():
            if family == "local":
                z = old.ecg.apply(y, features, dict(family="local",
                                                    alpha=parameters["alpha"],
                                                    shrink="firm"))
            else:
                z = old.gated(y, features, **parameters)
            rows.append(dict(record=rec, noise=kind, snr_db=snr, seed=seed,
                             input_sha256=meta["input_sha256"], method=name,
                             **score_case(x, y, masks, z)))
    print("validation record", rec, "rows", len(rows), flush=True)
expected = 20 * 12 * len(methods)
assert len(rows) == expected
pd.DataFrame(rows).to_csv(HERE / "validation_raw.csv", index=False)
official = {}
for line in (ROOT / "data" / "ludb" / "SHA256SUMS.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) == 2:
        official[parts[1]] = parts[0]
for rec in records:
    for ext in ["dat", "hea"]:
        name = f"data/{rec}.{ext}"
        assert sha(ROOT / "data" / "ludb" / f"{rec}.{ext}") == official[name]
write_json(HERE / "validation_metadata.json", dict(
    status="completed", records=records, inputs=20 * 12, methods=list(methods),
    rows=expected, official_file_hashes_verified=40,
    allocation_sha256=sha(HERE / "record_allocation.csv"),
    development_selection_sha256=sha(selection_path),
    frozen_config_sha256=sha(FROZEN_PATH),
    code_sha256=sha(Path(__file__)), python=sys.version,
))

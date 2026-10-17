"""Prospectively specified computational additions for the author revision."""
from pathlib import Path
import os, sys, json, hashlib, time, itertools, argparse, importlib.metadata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(HERE / "deps"))
sys.path.insert(1, str(ROOT / "work/reprodeps"))
os.environ["ECG_SCIENCE_LIB"] = str(ROOT / "work/reprodeps")
os.environ["MPLCONFIGDIR"] = str(HERE / "mplconfig")
sys.path.insert(0, str(ROOT / "revision_v2/code"))
import run_v2 as old
import numpy as np, pandas as pd, scipy, wfdb, pywt
from scipy import signal
from scipy.optimize import linear_sum_assignment
from wfdb.processing import xqrs_detect
from PyEMD import EMD

OUT = HERE / "analysis"
OUT.mkdir(exist_ok=True)
CANDIDATES = [(k, a) for k in (1, 2, 3) for a in (.125, .25, .5, 1., 2., 4.)]
METHODS = ["raw", "local_frozen", "gate_linear_subband", "gate_sqrt_subband", "emd_firm"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, x):
    (OUT / name).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def emd_features(y):
    yp = np.pad(np.asarray(y, dtype=float), (512, 512), mode="reflect")
    e = EMD(spline_kind="cubic", nbsym=2)
    e.emd(yp, max_imf=6)
    imfs, residual = e.get_imfs_and_residue()
    assert np.max(np.abs(imfs.sum(axis=0) + residual - yp)) < 1e-10
    scales = np.median(np.abs(imfs - np.median(imfs, axis=1, keepdims=True)), axis=1) / .67448975
    return yp, imfs, residual, scales


def emd_apply(y, f, k, alpha):
    yp, imfs, residual, scales = f
    z = yp.copy()
    for i in range(min(k, len(imfs))):
        tau = alpha * scales[i] * np.sqrt(2 * np.log(len(y)))
        z += old.ecg.shrink(imfs[i], tau, "firm") - imfs[i]
    return z[512:-512]


def select():
    score = np.zeros((len(CANDIDATES), 3))
    noise = old.rr.v.load_noise(["ma"])
    rows = []
    start = time.perf_counter()
    for clip in old.rr.v.load_clips(old.rr.CONFIG["development_records"]):
        for kind, db, seed in itertools.product(["awgn", "ma"], [0, 6, 12], [11, 22]):
            y, masks = old.rr.v.corrupt(clip, kind, "burst", db, seed, noise)
            f = emd_features(y)
            power = np.mean(clip["x"][masks["valid"]] ** 2)
            for i, (k, a) in enumerate(CANDIDATES):
                z = emd_apply(y, f, k, a)
                err = (z - clip["x"]) ** 2
                vals = [np.sqrt(np.mean(err[masks["active"]]) / power),
                        np.sqrt(np.mean(err[masks["quiet"]]) / power),
                        np.mean(err[masks["valid"]]) / power]
                score[i] += vals
                rows.append(dict(record=clip["record"], start=clip["start"], noise=kind,
                                 db=db, seed=seed, k=k, alpha=a,
                                 active_nrmse=vals[0], quiet_nrmse=vals[1], whole_nmse=vals[2]))
        print("EMD development", clip["record"], clip["start"], "elapsed", round(time.perf_counter()-start, 1), flush=True)
    count = len(rows) // len(CANDIDATES)
    assert count == 72
    score /= count
    cap = json.loads((ROOT/"revision_v2/results/frozen_config.json").read_text())["active_cap"]
    feasible = [i for i in range(len(CANDIDATES)) if score[i, 0] <= cap]
    ix = min(feasible, key=lambda i: (score[i, 1], score[i, 0], i)) if feasible else min(range(len(CANDIDATES)), key=lambda i: (score[i, 2], i))
    k, a = CANDIDATES[ix]
    pd.DataFrame(rows).to_csv(OUT/"emd_development_raw.csv", index=False)
    pd.DataFrame([dict(k=k, alpha=a, active_nrmse=s[0], quiet_nrmse=s[1], whole_nmse=s[2])
                  for (k, a), s in zip(CANDIDATES, score)]).to_csv(OUT/"emd_development_scores.csv", index=False)
    dump("emd_frozen.json", dict(k=k, alpha=a, candidate_count=len(CANDIDATES),
         feasible_count=len(feasible), fallback=not bool(feasible), active_cap=cap,
         active_nrmse=float(score[ix,0]), quiet_nrmse=float(score[ix,1]), whole_nmse=float(score[ix,2]),
         development_inputs=count, library="EMD-signal", version=importlib.metadata.version("EMD-signal"),
         padding=512, max_imf=6, spline_kind="cubic", nbsym=2,
         original_frozen_hash=sha(ROOT/"revision_v2/results/frozen_config.json"),
         plan_hash=sha(HERE/"ANALYSIS_PROTOCOL.md"), script_hash=sha(__file__),
         holdout_processed_during_selection=False))
    print("EMD FROZEN", k, a, "feasible", len(feasible), flush=True)


def qrs_metrics(ref, pred):
    ref, pred = np.asarray(ref, int), np.asarray(pred, int)
    if len(ref) and len(pred):
        distance = np.abs(ref[:,None] - pred[None,:])
        cost = np.where(distance <= 54, distance, 1000000)
        ri, pi = linear_sum_assignment(cost)
        ok = distance[ri,pi] <= 54
        errors = distance[ri[ok],pi[ok]] / 360 * 1000
    else:
        errors = np.array([])
    tp = len(errors); fp = len(pred)-tp; fn = len(ref)-tp
    return dict(tp=tp, fp=fp, fn=fn, ref_beats=len(ref), predicted_beats=len(pred),
                sensitivity=tp/len(ref) if len(ref) else None,
                ppv=tp/len(pred) if len(pred) else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                localization_mae_ms=float(np.mean(errors)) if len(errors) else None)


def predict(z):
    return xqrs_detect(z, fs=360, learn=True, verbose=False)


def evaluate():
    part = os.environ.get("ECG_REVISION_PART")
    cfg = json.loads((OUT/"emd_frozen.json").read_text())
    original = json.loads((ROOT/"revision_v2/results/frozen_config.json").read_text())["chosen"]
    saved = pd.read_csv(ROOT/"revision_v2/results/holdout_raw.csv")
    saved = saved[saved.noise.isin(["awgn", "ma1"])].set_index(["record", "noise", "db", "seed", "method"])
    inputs = pd.read_csv(ROOT/"revision_v2/results/holdout_inputs.csv").set_index(["record", "noise", "db", "seed"])
    old.noise_inputs.cache = old.noise_inputs()
    metrics, detections, references, checks = [], [], [], []
    start = time.perf_counter()
    for record_index, clip in enumerate(old.clips()):
        if part is not None and record_index % 4 != int(part):
            continue
        recfile = OUT/f"record_{clip['record']}.csv"
        detfile = OUT/f"detections_{clip['record']}.jsonl"
        if recfile.exists() and detfile.exists():
            print("reuse completed record", clip["record"], flush=True)
            continue
        valid_ref = clip["beats"][(clip["beats"] >= 360) & (clip["beats"] < len(clip["x"])-360)]
        refs = predict(clip["x"])
        refs = refs[(refs >= 360) & (refs < len(clip["x"])-360)]
        references.append(dict(record=int(clip["record"]), **qrs_metrics(valid_ref, refs)))
        localrows = []
        with detfile.open("w", encoding="utf-8") as df:
            for kind, db, seed in itertools.product(["awgn", "ma1"], [0, 6, 12], [6101,6102,6103]):
                y, masks, meta = old.corrupt(clip, kind, db, seed)
                keys = (int(clip["record"]), kind, db, seed)
                assert meta["input_sha256"] == inputs.loc[keys, "input_sha256"]
                f = old.ecg.features(y)
                ef = emd_features(y)
                outputs = dict(raw=y, local_frozen=old.ecg.apply(y, f, dict(family="local", alpha=3, shrink="firm")),
                               emd_firm=emd_apply(y, ef, cfg["k"], cfg["alpha"]))
                for name in ["gate_linear_subband", "gate_sqrt_subband"]:
                    outputs[name] = old.gated(y, f, **original[name])
                for method in METHODS:
                    z = outputs[method]
                    assert z.shape == y.shape and np.isfinite(z).all()
                    m = old.rr.full_metrics(clip, z, y, masks)
                    if method != "emd_firm":
                        prior = saved.loc[(*keys, method)]
                        discrepancy = max(abs(float(m[c])-float(prior[c])) for c in ["quiet_rmse_mv","active_rmse_mv","rmse_mv","prd_centered_pct"])
                        assert discrepancy < 1e-10, (keys, method, discrepancy)
                        checks.append(discrepancy)
                    p = predict(z)
                    p = p[(p >= 360) & (p < len(y)-360)]
                    q = qrs_metrics(valid_ref, p)
                    localrows.append(dict(record=keys[0], noise=kind, db=db, seed=seed, method=method,
                                          input_sha256=meta["input_sha256"], **m, **q))
                    df.write(json.dumps(dict(record=keys[0],noise=kind,db=db,seed=seed,method=method,
                                             reference_peaks=valid_ref.tolist(), predictions=p.tolist()), separators=(",",":"))+"\n")
        pd.DataFrame(localrows).to_csv(recfile, index=False)
        (OUT/f"reference_{clip['record']}.json").write_text(json.dumps(references[-1]), encoding="utf-8")
        print("holdout", clip["record"], "rows", len(localrows), "seconds", round(time.perf_counter()-start,1), flush=True)
    if part is not None:
        dump(f"partition_{part}_checks.json", dict(status="passed", partition=int(part),
             max_original_metric_difference=max(checks) if checks else 0., input_hashes_all_match=True))
        return
    combine()


def combine():
    allrows = pd.concat([pd.read_csv(OUT/f"record_{r}.csv") for r in range(2,180,3)], ignore_index=True)
    allrows.to_csv(OUT/"revision_holdout_raw.csv", index=False)
    pd.DataFrame([json.loads((OUT/f"reference_{r}.json").read_text()) for r in range(2,180,3)]).to_csv(OUT/"qrs_reference.csv", index=False)
    dump("evaluation_checks.json", dict(status="passed", records=60, inputs=1080, rows=len(allrows),
         input_hashes_all_match=True, compared_original_outputs=4320,
         max_original_metric_difference=max([json.loads(p.read_text())["max_original_metric_difference"] for p in OUT.glob("partition_*_checks.json")] or [0.]),
         emd_frozen_hash=sha(OUT/"emd_frozen.json"), plan_hash=sha(HERE/"ANALYSIS_PROTOCOL.md"),
         detector="WFDB XQRS", wfdb=wfdb.__version__, learn=True, matching_window_ms=150,
         scoring="1 <= time < 9 seconds", detector_tuned=False, analysis="post-review exploratory",
         python=sys.version, versions=dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, pywt=pywt.__version__),
         script_hash=sha(__file__)))


def summarize():
    df = pd.read_csv(OUT/"revision_holdout_raw.csv")
    olddf = pd.read_csv(ROOT/"revision_v2/results/holdout_raw.csv")
    olddf = olddf[olddf.noise.isin(["awgn", "ma1"])]
    ordered = ["raw","local_frozen","gate_linear_global","gate_sqrt_global","gate_linear_subband","gate_historical","nlm_tuned","gate_sqrt_subband","emd_firm"]
    joined = pd.concat([olddf, df[df.method=="emd_firm"]], ignore_index=True)
    cols = ["quiet_rmse_mv","active_rmse_mv","rmse_mv","prd_centered_pct"]
    per = joined.groupby(["record","method"])[cols].mean().reset_index()
    per.to_csv(OUT/"voltage_prd_record.csv", index=False)
    table = per.groupby("method")[cols].mean().reindex(ordered)
    table.to_csv(OUT/"voltage_prd_summary.csv")
    qcols = ["sensitivity","ppv","f1","localization_mae_ms"]
    qr = df.groupby(["record","method"])[qcols].mean().reset_index()
    qr.to_csv(OUT/"qrs_record_metrics.csv", index=False)
    rng = np.random.default_rng(20261017)
    samples = rng.integers(0,60,size=(20000,60))
    qsummary, contrasts = [], []
    pivot = qr.pivot(index="record",columns="method",values="f1")
    for method in METHODS:
        a = qr[qr.method==method].sort_values("record")
        row = dict(method=method, records=len(a))
        for c in qcols:
            values=a[c].to_numpy(); means=np.nanmean(values[samples],axis=1)
            lo,hi=np.quantile(means,[.025,.975]); row.update({c:float(np.nanmean(values)),c+"_low":float(lo),c+"_high":float(hi)})
        counts=df[df.method==method][["tp","fp","fn"]].sum()
        row.update({c:int(counts[c]) for c in counts.index})
        qsummary.append(row)
    for comparator in ["raw","local_frozen","gate_linear_subband","emd_firm"]:
        d=(pivot["gate_sqrt_subband"]-pivot[comparator]).to_numpy()*100
        lo,hi=np.quantile(d[samples].mean(axis=1),[.025,.975])
        contrasts.append(dict(method="gate_sqrt_subband",comparator=comparator,f1_difference_pp=float(d.mean()),ci95_low=float(lo),ci95_high=float(hi),records=60))
    pd.DataFrame(qsummary).to_csv(OUT/"qrs_summary.csv",index=False)
    pd.DataFrame(contrasts).to_csv(OUT/"qrs_contrasts.csv",index=False)
    noise_summary=df.groupby(["record","noise","method"])[qcols].mean().groupby(["noise","method"]).mean()
    noise_summary.to_csv(OUT/"qrs_by_noise.csv")
    refs=pd.read_csv(OUT/"qrs_reference.csv")
    dump("summary_checks.json",dict(records=60,conditions_per_record=18,methods=METHODS,
         qrs_metrics_aggregation="mean per input, then record, then equal record means",
         reference_beats_scored_once=int(refs.ref_beats.sum()),reference_signal_f1_mean=float(refs.f1.mean()),
         noisy_reference_events_per_method=int(df[df.method=="raw"].ref_beats.sum()),
         bootstrap_draws=20000,bootstrap_unit="ECG record",bootstrap_seed=20261017,
         prd_formula="100 * RMSE_valid / RMS_valid(reference), after whole-clip mean centering",
         prd_is_rescaling_of_whole_normalized_rmse=True,
         whole_prd_squared_check_max=float(np.max(np.abs(joined.prd_centered_pct - 100*joined.rmse_mv/joined.reference_rms_mv))),
         multiplicity_control=False,clinical_endpoint="QRS beat-detection task, not diagnosis"))
    print(table.to_string());print(pd.DataFrame(qsummary).to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))


def smoke():
    rng=np.random.default_rng(10172026)
    y=np.sin(np.arange(3600)*2*np.pi/360)+.1*rng.normal(size=3600)
    t=time.perf_counter();f=emd_features(y)
    assert np.max(np.abs(emd_apply(y,f,3,0)-y))<1e-12
    assert np.isfinite(emd_apply(y,f,3,.5)).all()
    assert qrs_metrics([360,720],[362,718,1000])["tp"]==2
    assert qrs_metrics([360,720],[])["fn"]==2
    assert qrs_metrics([360],[414])["tp"]==1
    assert qrs_metrics([360],[415])["tp"]==0
    print("Meaningful implementation checks passed; EMD seconds",time.perf_counter()-t,"IMFs",f[1].shape,flush=True)
    # Confirm the deployed detector can run against this SciPy version.
    clip=next(old.clips());p=predict(clip["x"])
    print("XQRS reference smoke",clip["record"],len(p),"annotated",len(clip["beats"]),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["smoke","select","evaluate","combine","summarize"]);a=p.parse_args()
    globals()[a.phase]()

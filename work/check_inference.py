from pathlib import Path
import sys,os,json,time,platform,tracemalloc
ROOT=Path(__file__).resolve().parents[1]
lib=Path(os.environ.get('ECG_SCIENCE_LIB',str(ROOT/'work/reprodeps')))
if lib.exists():sys.path.insert(0,str(lib))
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True
import numpy as np,ecg_denoise as e
rng=np.random.default_rng(1004);y=rng.normal(size=10800)
f=e.features(y)
identity=e.apply(y,f,dict(family='background',alpha=0,c=2,shrink='firm'))
uncal=e.apply(y,f,dict(family='background',alpha=3,c=0,shrink='firm'))
local=e.apply(y,f,dict(family='local',alpha=3,shrink='firm'))
assert np.max(np.abs(identity-y))<1e-10
assert np.array_equal(uncal,local)
z=e.denoise_ecg(np.zeros(3600));assert np.max(np.abs(z))==0
truth=np.r_[rng.normal(scale=.01,size=1800),rng.normal(scale=1,size=1800)]
split=e.best_variance_split(truth);assert abs(split-1800)<100
for invalid in [np.ones((2,360)),np.r_[np.ones(400),np.nan]]:
    try:e.denoise_ecg(invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid input accepted')
times=[]
for i in range(60):
    st=time.perf_counter();e.denoise_ecg(y);times.append((time.perf_counter()-st)*1000)
tracemalloc.start();e.denoise_ecg(y);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
result=dict(reconstruction_max_abs_error=float(np.max(np.abs(identity-y))),background_c0_equals_local=True,zero_signal_identity=True,known_change_location=split,median_ms_per_30s=float(np.median(times)),p95_ms_per_30s=float(np.quantile(times,.95)),python_tracemalloc_peak_mb=peak/1e6,memory_note='tracemalloc allocations only, not process RSS or all native library buffers',platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),inference_note='Offline whole-clip quantile; speed does not establish causal real-time capability')
(ROOT/'results'/'inference_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)

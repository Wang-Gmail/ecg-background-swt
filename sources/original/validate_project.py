"""First-stage ECG denoising experiment. Run from the workspace root.

python outputs/validate_project.py --phase tune
python outputs/validate_project.py --phase test
No reference ECG, noise labels or beat annotations enter the denoiser.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'work' / 'pythonlibs'))
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'work' / 'mplconfig'))
import numpy as np
import scipy
from scipy import signal, ndimage
import pywt
import wfdb

OUT = BASE / 'outputs'
DATA = BASE / 'work' / 'data'
FS = 360
N = 30 * FS
LEVEL = 4
WAVELET = 'sym4'
DEV = ['100', '101', '103']
TEST = ['105', '106', '109', '111', '113', '117', '121']
STARTS = [30, 150]
BEATS = set('NLRBAaJSVFejEnfQ?')

def load_clips(records):
    result = []
    for rec in records:
        path = str(DATA / 'mitdb' / rec)
        ann = wfdb.rdann(path, 'atr')
        for start in STARTS:
            lo = start * FS
            r = wfdb.rdrecord(path, sampfrom=lo, sampto=lo+N, channels=[0])
            x = r.p_signal[:, 0].astype(float)
            x -= np.mean(x)
            b = np.array([int(t-lo) for t,s in zip(ann.sample, ann.symbol)
                          if lo <= t < lo+N and s in BEATS], dtype=int)
            result.append({'record':rec,'start':start,'x':x,'beats':b})
    return result

def load_noise(names=('ma','em','bw')):
    return {name: wfdb.rdrecord(str(DATA / 'nstdb' / name)).p_signal[:, 0]
            for name in names}

def corrupt(clip, kind, schedule, db, seed, noise):
    # Development uses noise from first 8 minutes; held-out tests use after 10 min.
    rec = int(clip['record'])
    rng = np.random.default_rng(seed*10007 + rec*101 + clip['start']*17)
    if kind == 'clean':
        z = np.zeros(N)
    elif kind == 'awgn':
        z = rng.normal(size=N)
    else:
        origin = 0 if clip['record'] in DEV else 600*FS
        max_start = 8*60*FS-N
        offset = origin + int(rng.integers(0, max_start))
        z = noise[kind][offset:offset+N].copy()
        z -= np.mean(z)
    env = np.ones(N)
    if schedule in ['burst', 'short', 'long']:
        env[:] = 0
        if schedule == 'burst':
            intervals = [(6, 10), (18, 22)]
        elif schedule == 'short':
            intervals = [(6, 6.5), (18, 18.5)]
        else:
            intervals = [(4, 26)]
        jitter = float(rng.uniform(-0.7, 0.7))
        for a,b in intervals:
            a, b = a+jitter, b+jitter
            lo, hi = round(a*FS), round(b*FS)
            env[lo:hi] = 1
            # 100 ms cosine ramps avoid artificial impulses at injection boundary.
            ramp = min(round(.1*FS), (hi-lo)//4)
            v = .5 - .5*np.cos(np.linspace(0, np.pi, ramp))
            env[lo:lo+ramp] = v
            env[hi-ramp:hi] = v[::-1]
    active = env > .999
    if kind != 'clean':
        target_power = np.mean(clip['x'][active]**2) / 10**(db/10)
        z *= np.sqrt(target_power/(np.mean(z[active]**2)+1e-30))
        z *= env
    else:
        z[:] = 0
    valid = np.ones(N, dtype=bool)
    valid[:FS] = valid[-FS:] = False
    quiet = (env == 0) & valid
    active &= valid
    return clip['x']+z, {'valid':valid, 'active':active, 'quiet':quiet, 'env':env}

def mad(x, axis=None):
    center = np.median(x, axis=axis, keepdims=(axis is not None))
    return np.median(np.abs(x-center), axis=axis) / .6744897501960817

def features(y, window=1.0):
    # Reflect padding isolates periodic SWT boundaries. Approximation stays unchanged.
    padded = np.pad(y, (512,512), mode='reflect')
    pad_tail = (-len(padded)) % (2**LEVEL)
    padded = np.pad(padded, (0,pad_tail), mode='reflect')
    coeff = pywt.swt(padded, WAVELET, level=LEVEL)
    w = round(window*FS)
    hop = round(.25*FS)
    locations = np.arange(0, len(padded), hop)
    local, global_s, background = [], [], []
    for _, d in coeff:
        xp = np.pad(d, (w//2,w-w//2), mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(xp,w)[locations]
        sf = mad(windows,axis=1)
        local.append(np.interp(np.arange(len(padded)),locations,sf))
        global_s.append(float(mad(d[512:512+N])))
        background.append(float(np.quantile(sf[(locations>=512)&(locations<512+N)], .2)))
    return {'coeff':coeff,'local':local,'global':global_s,'background':background,
            'padded':padded}

def shrink(d, tau, mode):
    mag = np.abs(d)
    if mode == 'soft':
        return np.sign(d)*np.maximum(mag-tau,0)
    # Firm threshold with upper threshold 2*tau; exact identity at tau=0.
    return np.sign(d)*np.where(mag>=2*tau,mag,np.maximum(2*(mag-tau),0))

def apply_method(y, f, p):
    family = p['family']
    if family == 'garcia2018_adapt' and 'median2018' not in f:
        f['median2018'] = [ndimage.median_filter(np.abs(d),size=round(.2*FS)+1)
                           for _,d in f['coeff']]
    if family == 'raw':
        return y.copy()
    if family == 'lowpass':
        return signal.sosfiltfilt(signal.butter(4,p['cutoff'],fs=FS,output='sos'),y)
    if family == 'dwt':
        c = pywt.wavedec(y,WAVELET,level=LEVEL,mode='symmetric')
        c = [c[0]]+[shrink(d,p['alpha']*mad(d),p['shrink']) for d in c[1:]]
        return pywt.waverec(c,WAVELET,mode='symmetric')[:len(y)]
    result = []
    if family == 'gate2016_adapt':
        # Adaptation of Tejasri 2016 Eq 3.12-3.13 to SWT and firm/soft shrinkage.
        # This is explicitly not a numerical reproduction of the dissertation.
        if 'gamma2016' not in f:
            d1,d2 = f['coeff'][-1][1], f['coeff'][-2][1]
            envelope = ndimage.uniform_filter1d(ndimage.median_filter(
                (np.abs(d1)+np.abs(d2))/2,size=round(.2*FS)+1),size=round(.2*FS)+1)
            core=envelope[512:512+N]
            low,high = np.quantile(core,.05), .5*np.quantile(core,.95)
            f['gamma2016'] = np.clip((envelope-low)/max(high-low,1e-12),0,1)
        gamma = f['gamma2016']
    for i,(a,d) in enumerate(f['coeff']):
        if family == 'global':
            tau = p['alpha']*f['global'][i]
        elif family == 'local':
            tau = p['alpha']*f['local'][i]
        elif family == 'background':
            tau = p['alpha']*np.sqrt(np.maximum(
                f['local'][i]**2-(p['c']*f['background'][i])**2,0))
        elif family == 'gate2016_adapt':
            tau = p['alpha']*gamma*np.quantile(np.abs(d[512:512+N]),p['quantile'])
        elif family == 'garcia2018_adapt':
            tau = p['alpha']*f['median2018'][i]
        else:
            raise ValueError(family)
        if family == 'garcia2018_adapt':
            dnew=np.where(np.abs(d)>1.5*tau,d,shrink(d,tau,'soft'))
        else:
            dnew=shrink(d,tau,p['shrink'])
        result.append((a,dnew))
    return pywt.iswt(result,WAVELET)[512:512+len(y)]

def evaluate(x,y,masks,beats):
    valid = masks['valid']
    err = (y-x)**2
    mse = float(np.mean(err[valid]))
    power = float(np.mean(x[valid]**2))
    result = {'rmse_mv':float(np.sqrt(mse)), 'snr_out_db':float(10*np.log10(power/max(mse,1e-30))),
              'prd_centered_pct':float(100*np.sqrt(mse/max(power,1e-30)))}
    for label in ['active','quiet']:
        mask = masks[label]
        result[f'{label}_rmse_mv'] = float(np.sqrt(np.mean(err[mask]))) if mask.any() else None
    # Fixed annotated sample error assesses morphology; it is not a detection score.
    b = beats[(beats>=FS)&(beats<N-FS)]
    result['beat_sample_mae_mv'] = float(np.mean(np.abs(y[b]-x[b]))) if len(b) else None
    qrs = np.zeros(N,dtype=bool)
    for t in b:
        qrs[max(0,t-round(.06*FS)):min(N,t+round(.06*FS)+1)] = True
    result['qrs_window_rmse_mv'] = float(np.sqrt(np.mean(err[qrs&valid])))
    quiet_qrs=qrs&masks['quiet']
    quiet_beats=b[masks['quiet'][b]]
    result['quiet_qrs_rmse_mv']=float(np.sqrt(np.mean(err[quiet_qrs]))) if quiet_qrs.any() else None
    result['quiet_beat_sample_mae_mv']=float(np.mean(np.abs(y[quiet_beats]-x[quiet_beats]))) if len(quiet_beats) else None
    return result

def parameter_grid():
    alpha = [.25,.5,.75,1,1.5,2,3,4,6,8]
    grid = {'lowpass':[{'family':'lowpass','cutoff':v} for v in [25,35,45,60,80]],
            'dwt':[], 'global':[], 'local':[], 'gate2016_adapt':[], 'background':[],
            'garcia2018_adapt':[{'family':'garcia2018_adapt','alpha':a} for a in alpha]}
    for mode in ['firm','soft']:
        for a in alpha:
            for fam in ['dwt','global','local']:
                grid[fam].append({'family':fam,'alpha':a,'shrink':mode})
            for q in [.5,.75,.9]:
                grid['gate2016_adapt'].append({'family':'gate2016_adapt','alpha':a,'shrink':mode,'quantile':q})
            for c in [0,.5,.75,1,1.25,1.5,2]:
                grid['background'].append({'family':'background','alpha':a,'shrink':mode,'c':c})
    return grid

def tune():
    clips = load_clips(DEV)
    noise = load_noise(['ma'])
    grid = parameter_grid()
    sums = {name:np.zeros(len(params)) for name,params in grid.items()}
    n=0
    started=time.perf_counter()
    for clip in clips:
        for kind in ['awgn','ma']:
            for schedule in ['burst','stationary']:
                for db in [0,6,12]:
                    for seed in [11,22]:
                        y,m=corrupt(clip,kind,schedule,db,seed,noise)
                        f=features(y)
                        for name, params in grid.items():
                            for i,p in enumerate(params):
                                yhat=apply_method(y,f,p)
                                # Each case equal weight, center-normalized MSE.
                                sums[name][i]+=np.mean((yhat[m['valid']]-clip['x'][m['valid']])**2)/np.mean(clip['x'][m['valid']]**2)
                        n+=1
        print(f'tuned development record {clip["record"]} start={clip["start"]}; cases={n}',flush=True)
    best = {name:grid[name][int(np.argmin(scores))] for name,scores in sums.items()}
    # Paired ablation keeps proposed alpha/shrink; only c=0 cancels background correction.
    best['ablation_no_background'] = dict(best['background'],family='local')
    best['raw']={'family':'raw'}
    config={'development_records':DEV,'heldout_records':TEST,'starts_s':STARTS,'duration_s':30,
            'fs':FS,'wavelet':WAVELET,'level':LEVEL,'local_window_s':1,'hop_s':.25,'background_quantile':.2,
            'selection_objective':'mean reference-power normalized MSE, equal case weights',
            'development_cases':n,'best':best,'grid':grid,
            'validation_scores':{name:(v/n).tolist() for name,v in sums.items()},
            'tuning_seconds':time.perf_counter()-started}
    path=OUT/'frozen_config.json'
    path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({'best':best,'cases':n,'seconds':config['tuning_seconds'],
                     'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2),flush=True)

def test():
    config_path=OUT/'frozen_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    clips=load_clips(TEST)
    noise=load_noise()
    rows=[]
    timings={name:[] for name in config['best']}
    example=None
    t0=time.perf_counter()
    for clip in clips:
        scenarios=[(k,s,d,seed) for k in ['awgn','ma','em','bw'] for s in ['burst','stationary']
                   for d in [0,6,12] for seed in [101,202,303]]
        scenarios += [('ma',s,6,seed) for s in ['short','long'] for seed in [101,202,303]]
        scenarios += [('clean','stationary',0,101)]
        for kind,schedule,db,seed in scenarios:
            y,m=corrupt(clip,kind,schedule,db,seed,noise)
            ft=time.perf_counter(); f=features(y); feature_ms=(time.perf_counter()-ft)*1000
            this={}
            for name,p in config['best'].items():
                st=time.perf_counter(); z=apply_method(y,f,p); ms=(time.perf_counter()-st)*1000
                if p['family'] in ['local','background','gate2016_adapt','global','garcia2018_adapt']: ms+=feature_ms
                timings[name].append(ms)
                row={'record':clip['record'],'start_s':clip['start'],'noise':kind,'schedule':schedule,
                     'target_active_snr_db':db,'seed':seed,'method':name}
                row.update(evaluate(clip['x'],z,m,clip['beats']))
                raw_mse=np.mean((y[m['valid']]-clip['x'][m['valid']])**2)
                row['snr_gain_db']=None if kind=='clean' else float(10*np.log10(max(raw_mse,1e-30)/max(row['rmse_mv']**2,1e-30)))
                rows.append(row)
                this[name]=z
            if clip['record']=='105' and clip['start']==30 and kind=='ma' and schedule=='burst' and db==6 and seed==101:
                example={'x':clip['x'],'y':y,'env':m['env'],**this}
        print(f'tested record {clip["record"]} start={clip["start"]}; rows={len(rows)}',flush=True)
    runtime={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
             'pywt':pywt.__version__,'wfdb':wfdb.__version__,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
             'experiment_seconds':time.perf_counter()-t0,
             'median_processing_ms_per_30s':{k:float(np.median(v)) for k,v in timings.items()},
             'timing_note':'includes shared full feature extraction for SWT methods; conservative, no standalone optimization'}
    (OUT/'experiment_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'example_waveforms.npz',**example)
    print(json.dumps(runtime,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['tune','test'],required=True)
    args=parser.parse_args()
    {'tune':tune,'test':test}[args.phase]()

from pathlib import Path
import sys,os,json,hashlib,time,argparse,itertools
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'revision_v2';OUT=BASE/'results';OUT.mkdir(exist_ok=True)
os.environ['ECG_SCIENCE_LIB']=str(ROOT/'work/reprodeps');os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(ROOT/'work'));sys.path.insert(0,str(BASE/'code'))
import run_research as rr
from methods_v2 import gated,nlm_many,nlm_literal
np,pd,signal,wfdb=rr.np,rr.pd,rr.signal,rr.wfdb
ecg=rr.ecg
PROTOCOL=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
ADDENDUM=json.loads((BASE/'implementation_addendum.json').read_text(encoding='utf-8'))
FAMILIES={f'gate_{shape}_{source}':dict(source=source,shape=shape) for source in ['global','subband'] for shape in ['linear','sqrt']}
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'protocol.json',BASE/'implementation_addendum.json',BASE/'code/methods_v2.py']}

def verify():
    rng=np.random.default_rng(10152026);checks={}
    y=rng.normal(size=144)
    for lam in [.15,.6,2.]:
        fast=nlm_many(y,[lam],search=7,half=3)[0];ref=nlm_literal(y,lam,search=7,half=3)
        checks[f'nlm_patch_oracle_{lam}']=float(np.max(np.abs(fast-ref)))
        assert np.max(np.abs(fast-ref))<1e-12
    y=rng.normal(size=3600);f=ecg.features(y)
    a=gated(y,f,'subband','sqrt',3,2);b=ecg.denoise_ecg(y)
    checks['original_background_parity']=float(np.max(np.abs(a-b)));assert np.array_equal(a,b)
    for source,shape in itertools.product(['global','subband'],['linear','sqrt']):
        z=gated(y,f,source,shape,3,0);local=ecg.apply(y,f,dict(family='local',alpha=3,shrink='firm'))
        checks[f'zero_c_{source}_{shape}']=float(np.max(np.abs(z-local)));assert np.max(np.abs(z-local))<1e-12
    z=nlm_many(np.ones(100),[.3],search=7,half=3)[0]
    assert np.max(np.abs(z-1))<1e-12
    dump(OUT/'implementation_checks.json',dict(status='passed',checks=checks,source_hashes=hashes(),author_matlab_runtime_parity=False,validation='independent direct patch-sum weights and invariants'))
    print('Meaningful implementation checks passed',checks,flush=True)

def development():
    candidates=[]
    for name,base in FAMILIES.items():
        for a,c in itertools.product(PROTOCOL['matched_factorial']['alpha_grid'],PROTOCOL['matched_factorial']['c_grid']):candidates.append((name,dict(**base,alpha=a,c=c)))
    for a in PROTOCOL['matched_factorial']['local_reference_alpha_grid']:candidates.append(('local',dict(alpha=a)))
    kappas=ADDENDUM['strong_external_comparator']['kappa_candidates']
    for k in kappas:candidates.append(('nlm',dict(kappa=k)))
    scores=np.zeros((len(candidates),3));count=0;noise=rr.v.load_noise(['ma'])
    for clip in rr.v.load_clips(rr.CONFIG['development_records']):
        for kind,db,seed in itertools.product(['awgn','ma'],[0,6,12],[11,22]):
            y,m=rr.v.corrupt(clip,kind,'burst',db,seed,noise);f=ecg.features(y)
            scale=float(np.quantile(f['local'][-1],.9));nlm=nlm_many(y,[k*scale for k in kappas])
            power=np.mean(clip['x'][m['valid']]**2)
            for i,(name,p) in enumerate(candidates):
                if name in FAMILIES:z=gated(y,f,**p)
                elif name=='local':z=ecg.apply(y,f,dict(family='local',alpha=p['alpha'],shrink='firm'))
                else:z=nlm[kappas.index(p['kappa'])]
                e=(z-clip['x'])**2
                scores[i]+=[np.sqrt(np.mean(e[m['active']])/power),np.sqrt(np.mean(e[m['quiet']])/power),np.mean(e[m['valid']])/power]
            count+=1
        print('Development clip',clip['record'],clip['start'],'cases',count,flush=True)
    scores/=count;cap=1.01*min(scores[i,0] for i,(name,_) in enumerate(candidates) if name=='local')
    chosen={};selection={}
    for method in [*FAMILIES,'nlm']:
        ix=[i for i,(name,p) in enumerate(candidates) if name==method];feasible=[i for i in ix if scores[i,0]<=cap]
        best=min(feasible,key=lambda i:(scores[i,1],scores[i,0],i)) if feasible else min(ix,key=lambda i:(scores[i,2],i))
        chosen[method]=candidates[best][1];selection[method]=dict(feasible=bool(feasible),candidates=len(ix),active_nrmse=float(scores[best,0]),quiet_nrmse=float(scores[best,1]),whole_nmse=float(scores[best,2]))
    pd.DataFrame([dict(method=name,parameters=json.dumps(p),active_nrmse=s[0],quiet_nrmse=s[1],whole_nmse=s[2]) for (name,p),s in zip(candidates,scores)]).to_csv(OUT/'development_scores.csv',index=False)
    dump(OUT/'frozen_config.json',dict(chosen=chosen,selection=selection,active_cap=float(cap),development_cases=count,source_hashes=hashes(),holdout_seen_during_selection=False))
    print('FROZEN',chosen,selection,flush=True)

def clips():
    for rec in range(2,180,3):
        path=str(BASE/'data/ludb_holdout'/str(rec));record=wfdb.rdrecord(path,channels=[1]);assert record.fs==500 and record.sig_name[0].lower()=='ii'
        x=signal.resample_poly(record.p_signal[:,0],18,25);x-=x.mean();ann=wfdb.rdann(path,'ii')
        beats=np.array([round(s*18/25) for s,sym in zip(ann.sample,ann.symbol) if sym=='N'],dtype=int)
        yield dict(record=str(rec),start=0,x=x,beats=beats,waves=rr.wave_masks(ann,len(x)))

def noise_inputs():
    n=wfdb.rdrecord(str(ROOT/'data/nstdb/ma'));assert n.fs==360 and n.p_signal.shape[1]>=2
    return dict(ma1=n.p_signal[:,1])

def corrupt(clip,kind,db,seed,fraction=None,bursts=None):
    n=len(clip['x']);fs=360;rng=np.random.default_rng(seed*10007+int(clip['record'])*101)
    env=np.zeros(n)
    if fraction is None:
        jitter=float(rng.uniform(-.15,.15));intervals=[(2+jitter,3.3+jitter),(6+jitter,7.3+jitter)]
    elif fraction==1:env[:]=1;intervals=[]
    else:
        cell=10/bursts;intervals=[((i+.5)*cell-fraction*cell/2,(i+.5)*cell+fraction*cell/2) for i in range(bursts)]
    for a,b in intervals:
        lo,hi=round(a*fs),round(b*fs);env[lo:hi]=1
        ramp=min(36,(hi-lo)//4);edge=.5-.5*np.cos(np.linspace(0,np.pi,ramp));env[lo:lo+ramp]=edge;env[hi-ramp:hi]=edge[::-1]
    if kind=='ma1':
        pool=noise_inputs.cache['ma1'];lo=1080*fs;hi=1780*fs-n;offset=int(rng.integers(lo,hi));z=pool[offset:offset+n].copy();z-=z.mean()
    else:
        offset=None;z=rng.normal(size=n)
        if kind=='blgn':z=signal.sosfiltfilt(signal.butter(4,[35,150],btype='bandpass',fs=fs,output='sos'),z)
    active=env>.999;assert active.any()
    z*=np.sqrt(np.mean(clip['x'][active]**2)/(10**(db/10)*max(np.mean(z[active]**2),1e-30)));z*=env
    valid=np.ones(n,dtype=bool);valid[:fs]=False;valid[-fs:]=False
    boundary=np.zeros(n,dtype=bool)
    for t in np.where(np.diff((env>0).astype(int))!=0)[0]:boundary[max(0,t-72):min(n,t+73)]=True
    masks=dict(valid=valid,active=active&valid,quiet=(env==0)&valid,env=env,transition=(env>0)&(env<.999)&valid,boundary=boundary&valid)
    y=clip['x']+z
    metadata=dict(input_sha256=hashlib.sha256(y.tobytes()).hexdigest(),noise_offset_samples=offset,positive_fraction=float(np.mean(env>0)),valid_positive_fraction=float(np.mean(env[valid]>0)),quiet_samples=int(masks['quiet'].sum()),active_samples=int(masks['active'].sum()))
    return y,masks,metadata

def evaluate(phase):
    frozen=json.loads((OUT/'frozen_config.json').read_text(encoding='utf-8'));assert frozen['source_hashes']==hashes()
    noise_inputs.cache=noise_inputs();rows=[];inputs=[];t0=time.perf_counter();timings=[];example=None
    selected=list(clips());selected=selected if phase=='holdout' else selected[::3]
    for clip in selected:
        if phase=='holdout':conditions=[(kind,db,seed,None,None) for kind,db,seed in itertools.product(['awgn','ma1','blgn'],[0,6,12],[6101,6102,6103])]
        else:conditions=[(kind,6,seed,q,k) for kind,seed,q,k in itertools.product(['awgn','ma1'],[7401,7402],[.1,.25,.4,.6,.8,1.],[1,4]) if q<1 or k==1]
        for kind,db,seed,q,k in conditions:
            y,m,meta=corrupt(clip,kind,db,seed,q,k);f=ecg.features(y)
            scale=float(np.quantile(f['local'][-1],.9));kappa=frozen['chosen']['nlm']['kappa']
            unique_kappas=list(dict.fromkeys([kappa,.6]))
            st=time.perf_counter();nlm_unique=nlm_many(y,[value*scale for value in unique_kappas]);nlmtime=time.perf_counter()-st
            nlm=[nlm_unique[unique_kappas.index(kappa)],nlm_unique[unique_kappas.index(.6)]]
            outputs={'raw':lambda:y.copy(),'local_frozen':lambda:ecg.apply(y,f,dict(family='local',alpha=3,shrink='firm')),'background_original':lambda:ecg.denoise_ecg(y),'gate_historical':lambda:ecg.historical_gate(y),'gate_common_old':lambda:ecg.apply(y,f,dict(family='gate_common',alpha=3,ratio=1.,shrink='firm')),'lowpass_old':lambda:ecg.apply(y,f,dict(family='lowpass',cutoff=45)),'nlm_tuned':lambda:nlm[0],'nlm_fixed_0.6':lambda:nlm[1]}
            for name in FAMILIES:outputs[name]=lambda name=name:gated(y,f,**frozen['chosen'][name])
            for name,base in FAMILIES.items():outputs[name.replace('gate_','fixed_')]=lambda base=base:gated(y,f,**base,alpha=3,c=2)
            names=list(outputs);rotate=len(inputs)%len(names);names=names[rotate:]+names[:rotate]
            keys=dict(record=clip['record'],noise=kind,db=db,seed=seed,nominal_fraction=q,bursts=k)
            inputs.append(dict(**keys,**meta))
            zs={}
            for name in names:
                st=time.perf_counter();z=outputs[name]();duration=(time.perf_counter()-st)*1000
                assert np.isfinite(z).all() and z.shape==y.shape
                metrics=rr.full_metrics(clip,z,y,m)
                # Empty regions must remain missing; legacy evaluator can emit warnings.
                rows.append(dict(**keys,method=name,**metrics));timings.append(dict(**keys,method=name,apply_ms=duration,nlm_shared_two_bandwidth_ms=nlmtime*1000 if name.startswith('nlm') else None))
                zs[name]=z
            if example is None and kind=='awgn' and db==6:
                example=dict(reference=clip['x'],observed=y,envelope=m['env'],**zs)
        print(phase,'record',clip['record'],'inputs',len(inputs),'rows',len(rows),flush=True)
    pd.DataFrame(rows).to_csv(OUT/f'{phase}_raw.csv',index=False)
    pd.DataFrame(inputs).to_csv(OUT/f'{phase}_inputs.csv',index=False)
    pd.DataFrame(timings).to_csv(OUT/f'{phase}_timing.csv',index=False)
    np.savez_compressed(OUT/f'{phase}_example.npz',**example)
    dump(OUT/f'{phase}_metadata.json',dict(phase=phase,records=len(selected),inputs=len(inputs),rows=len(rows),seconds=time.perf_counter()-t0,source_hashes=hashes(),frozen_config_sha256=hashlib.sha256((OUT/'frozen_config.json').read_bytes()).hexdigest(),python=sys.version))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','development','holdout','stress']);args=p.parse_args()
    if args.phase=='verify':verify()
    elif args.phase=='development':development()
    else:evaluate(args.phase)

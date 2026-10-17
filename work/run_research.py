from pathlib import Path
import sys, os, json, hashlib, time, argparse, importlib.util
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
SCIENCE_LIB=Path(os.environ.get('ECG_SCIENCE_LIB',str(ROOT/'work/reprodeps')))
if SCIENCE_LIB.exists():sys.path.insert(0,str(SCIENCE_LIB))
sys.path.insert(0,str(ROOT/'src'))
os.environ['MPLCONFIGDIR']=str(ROOT/'work'/'mplconfig')
import numpy as np, pandas as pd, scipy, wfdb, pywt
from scipy import signal,ndimage
import ecg_denoise as ecg
spec=importlib.util.spec_from_file_location('legacy_ecg',ROOT/'sources'/'original'/'validate_project.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
v.DATA=ROOT/'data';v.OUT=ROOT/'results'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CONFIG=json.loads((ROOT/'sources'/'original'/'quiet_frozen_config.json').read_text(encoding='utf-8'))
PROTOCOL=ROOT/'configs'/'extension_protocol.json'

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def audit():
    old=pd.DataFrame(json.loads((ROOT/'sources'/'original'/'quiet_results.json').read_text(encoding='utf-8')))
    primary=old[(old.schedule=='burst')&old.noise.isin(['awgn','ma'])]
    totals=primary.groupby('method')[['quiet_rmse_mv','active_rmse_mv','rmse_mv','snr_out_db']].mean()
    changes={col:float(100*(totals.loc['background',col]/totals.loc['local',col]-1)) for col in ['quiet_rmse_mv','active_rmse_mv','rmse_mv']}
    changes['snr_out_db_delta']=float(totals.loc['background','snr_out_db']-totals.loc['local','snr_out_db'])
    diffs=[];parity=[];noise=v.load_noise()
    for clip in v.load_clips(CONFIG['confirmation_records']):
        for kind in ['awgn','ma']:
            for db in [0,6,12]:
                for seed in [101,202,303]:
                    y,m=v.corrupt(clip,kind,'burst',db,seed,noise)
                    f=ecg.features(y);z=ecg.apply(y,f,CONFIG['best']['background'])
                    zlegacy=v.apply_method(y,v.features(y),CONFIG['best']['background'])
                    parity.append(float(np.max(np.abs(z-zlegacy))))
                    metrics=v.evaluate(clip['x'],z,m,clip['beats'])
                    prior=primary[(primary.record==clip['record'])&(primary.start_s==clip['start'])&(primary.noise==kind)&(primary.target_active_snr_db==db)&(primary.seed==seed)&(primary.method=='background')]
                    assert len(prior)==1
                    diffs.append(max(abs(metrics[k]-float(prior.iloc[0][k])) for k in ['rmse_mv','quiet_rmse_mv','active_rmse_mv','snr_out_db']))
    assert max(diffs)<1e-8 and max(parity)<1e-10
    assert not(set(CONFIG['development_records'])&set(CONFIG['confirmation_records']))
    manifest=json.loads((ROOT/'sources'/'import_manifest.json').read_text(encoding='utf-8'))
    for e in manifest:assert hashlib.sha256((ROOT/e['local']).read_bytes()).hexdigest()==e['sha256']
    dump(OUT/'historical_audit.json',dict(primary_cases=len(primary[primary.method=='background']),max_saved_metric_discrepancy=max(diffs),max_inference_parity_error=max(parity),changes=changes,checked_import_files=len(manifest),grouped_means=totals.reset_index().to_dict(orient='records')))
    print('Audit passed',changes,flush=True)

def dev_tune():
    alphas=[.25,.5,.75,1,1.5,2,3,4,6,8]
    candidates=[]
    for mode in ['firm','soft']:
        candidates += [('local_tuned',{'family':'local','alpha':a,'shrink':mode}) for a in alphas]
    candidates += [('gate_common_tuned',{'family':'gate_common','alpha':a,'ratio':r,'shrink':'firm'}) for a in alphas for r in [1.,1.5,2.]]
    candidates += [('idt_common_tuned',{'family':'idt_common','alpha':a,'shrink':m}) for a in [.25,.5,1,2,3] for m in ['hard','soft']]
    candidates += [('idt_native_tuned',{'family':'idt_native','alpha':a,'shrink':m}) for a in [.25,.5,1,2,3] for m in ['hard','soft']]
    candidates += [('lowpass_tuned',{'family':'lowpass','cutoff':a}) for a in [25,35,45,60,80]]
    totals=np.zeros((len(candidates),3));n=0;noise=v.load_noise(['ma'])
    for clip in v.load_clips(CONFIG['development_records']):
        for kind in ['awgn','ma']:
            for db in [0,6,12]:
                for seed in [11,22]:
                    y,m=v.corrupt(clip,kind,'burst',db,seed,noise);f=ecg.features(y)
                    cn=ecg.idt_cache(y);cc=ecg.idt_cache(y,'sym4',4)
                    power=np.mean(clip['x'][m['valid']]**2)
                    for i,(name,p) in enumerate(candidates):
                        if p['family'].startswith('idt'):z=ecg.idt_apply(cn if p['family']=='idt_native' else cc,p['alpha'],p['shrink'])
                        else:z=ecg.apply(y,f,p)
                        err=(z-clip['x'])**2
                        totals[i]+=[np.sqrt(np.mean(err[m['active']])/power),np.sqrt(np.mean(err[m['quiet']])/power),np.mean(err[m['valid']])/power]
                    n+=1
        print('Development comparison',clip['record'],clip['start'],flush=True)
    totals/=n
    local_ids=[i for i,(name,p) in enumerate(candidates) if name=='local_tuned']
    cap=1.01*min(totals[i,0] for i in local_ids)
    chosen={};rules={};scores=[]
    for family in dict(candidates):
        ids=[i for i,(name,p) in enumerate(candidates) if name==family]
        feasible=[i for i in ids if totals[i,0]<=cap]
        selected=min(feasible,key=lambda i:totals[i,1]) if feasible else min(ids,key=lambda i:totals[i,2])
        chosen[family]=candidates[selected][1]
        rules[family]={'feasible':bool(feasible),'candidates':len(ids),'active_nrmse':float(totals[selected,0]),'quiet_nrmse':float(totals[selected,1]),'whole_nmse':float(totals[selected,2])}
    for (name,p),score in zip(candidates,totals):scores.append(dict(method=name,parameters=json.dumps(p),active_nrmse=score[0],quiet_nrmse=score[1],whole_nmse=score[2]))
    pd.DataFrame(scores).to_csv(OUT/'development_comparison_scores.csv',index=False)
    chosen['background_frozen']=CONFIG['best']['background']
    chosen['local_frozen']=CONFIG['best']['local']
    chosen['gate_historical_frozen']={'family':'gate_historical','k':1.2,'ratio':1.5}
    chosen['idt_native_original']={'family':'idt_native','alpha':1.,'shrink':'hard'}
    chosen['raw']={'family':'raw'}
    chosen['local_weak_alpha_2']={'family':'local','alpha':2,'shrink':'firm'}
    chosen['local_weak_alpha_1']={'family':'local','alpha':1,'shrink':'firm'}
    dump(OUT/'comparison_frozen_config.json',dict(best=chosen,selection=rules,dev_cases=n,active_cap=cap,protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),note='No extension data used in this selection; IDT has no verified author-code parity'))
    print('Frozen comparisons',chosen,rules,flush=True)

def wave_masks(ann,n):
    masks={k:np.zeros(n,dtype=bool) for k in ['p','qrs','t']};pending=None;label=None
    for s,sym in zip(ann.sample,ann.symbol):
        t=min(n-1,int(round(s*18/25)))
        if sym=='(':pending=t;label=None
        elif sym in ['p','N','t']:label={'p':'p','N':'qrs','t':'t'}[sym]
        elif sym==')' and pending is not None and label is not None:
            masks[label][pending:t+1]=True;pending=None;label=None
    return masks

def ludb_clips():
    for rec in range(1,179,3):
        path=str(ROOT/'data'/'ludb_selected'/str(rec));r=wfdb.rdrecord(path,channels=[1])
        assert r.fs==500 and r.sig_name[0].lower()=='ii'
        x=signal.resample_poly(r.p_signal[:,0],18,25);x-=np.mean(x)
        ann=wfdb.rdann(path,'ii')
        beats=np.asarray([round(s*18/25) for s,sym in zip(ann.sample,ann.symbol) if sym=='N'],dtype=int)
        yield dict(record=str(rec),start=0,x=x,beats=beats,waves=wave_masks(ann,len(x)))

def contaminate(clip,kind,schedule,db,seed,noise):
    n=len(clip['x']);fs=360
    rng=np.random.default_rng(seed*10007+int(clip['record'])*101)
    env=np.ones(n)
    if schedule in ['burst','short','long']:
        env[:]=0
        intervals={'burst':[(2,3.3),(6,7.3)],'short':[(2,2.5),(6,6.5)],'long':[(1.2,8.8)]}[schedule]
        jitter=float(rng.uniform(-.15,.15))
        for a,b in intervals:
            lo,hi=round((a+jitter)*fs),round((b+jitter)*fs);env[lo:hi]=1
            ramp=min(36,(hi-lo)//4);r=.5-.5*np.cos(np.linspace(0,np.pi,ramp));env[lo:lo+ramp]=r;env[hi-ramp:hi]=r[::-1]
    active=env>.999
    if kind=='clean':z=np.zeros(n);env[:]=0;active[:]=False
    else:
        if kind=='awgn':z=rng.normal(size=n)
        else:
            lo=1080*fs;hi=1780*fs-n;offset=int(rng.integers(lo,hi));z=noise[kind][offset:offset+n].copy();z-=np.mean(z)
        z*=np.sqrt(np.mean(clip['x'][active]**2)/(10**(db/10)*max(np.mean(z[active]**2),1e-30)));z*=env
    valid=np.ones(n,dtype=bool);valid[:fs]=False;valid[-fs:]=False
    boundary=np.zeros(n,dtype=bool)
    for t in np.where(np.diff((env>0).astype(int))!=0)[0]:boundary[max(0,t-72):min(n,t+73)]=True
    return clip['x']+z,dict(valid=valid,active=active&valid,quiet=(env==0)&valid,env=env,transition=(env>0)&(env<.999)&valid,boundary=boundary&valid)

def full_metrics(clip,z,y,m):
    # Legacy evaluation has fixed globals; set current length before evaluation.
    v.N=len(z);r=v.evaluate(clip['x'],z,m,clip['beats']);e=(z-clip['x'])**2
    scale=np.sqrt(np.mean(clip['x'][m['valid']]**2));r['reference_rms_mv']=float(scale)
    for key in ['quiet','active']:r[key+'_nrmse']=r[key+'_rmse_mv']/scale if r[key+'_rmse_mv'] is not None else None
    for key in ['transition','boundary']:
        r[key+'_rmse_mv']=float(np.sqrt(np.mean(e[m[key]]))) if m[key].any() else None
    for key,mask in clip.get('waves',{}).items():
        mask=mask&m['quiet'];r['quiet_'+key+'_full_rmse_mv']=float(np.sqrt(np.mean(e[mask]))) if mask.any() else None
    r['snr_gain_db']=float(10*np.log10(max(np.mean((y[m['valid']]-clip['x'][m['valid']])**2),1e-30)/max(r['rmse_mv']**2,1e-30))) if np.any(y!=clip['x']) else None
    return r

def run_set(database):
    conf=json.loads((OUT/'comparison_frozen_config.json').read_text(encoding='utf-8'))
    noise=v.load_noise();rows=[];runtimes=[];example=None;t0=time.perf_counter()
    if database=='ludb':clips=ludb_clips()
    else:v.N=30*360;clips=v.load_clips(CONFIG['confirmation_records'])
    for clip in clips:
        n=len(clip['x']);v.N=n
        scenarios=[(k,'burst',db,s) for k in ['awgn','ma'] for db in [0,6,12] for s in ([4101,4102,4103] if database=='ludb' else [101,202,303])]
        if database=='ludb':scenarios +=[(k,s,6,4101) for k,s in [('awgn','stationary'),('ma','stationary'),('ma','short'),('ma','long'),('em','burst'),('bw','burst'),('clean','clean')]]
        for kind,schedule,db,seed in scenarios:
            if database=='ludb':y,m=contaminate(clip,kind,schedule,db,seed,noise)
            else:
                y,m=v.corrupt(clip,kind,schedule,db,seed,noise);m['transition']=(m['env']>0)&(m['env']<.999)&m['valid'];m['boundary']=np.zeros(n,dtype=bool)
                for t in np.where(np.diff((m['env']>0).astype(int))!=0)[0]:m['boundary'][max(0,t-72):min(n,t+73)]=True
                m['boundary'] &=m['valid']
            st=time.perf_counter();f=ecg.features(y);feature_ms=(time.perf_counter()-st)*1000
            st=time.perf_counter();cn=ecg.idt_cache(y);native_ms=(time.perf_counter()-st)*1000
            st=time.perf_counter();cc=ecg.idt_cache(y,'sym4',4);common_ms=(time.perf_counter()-st)*1000
            zs={}
            for name,p in conf['best'].items():
                st=time.perf_counter();family=p['family'];shared=0
                if family.startswith('idt'):
                    z=ecg.idt_apply(cn if family=='idt_native' else cc,p['alpha'],p['shrink']);shared=native_ms if family=='idt_native' else common_ms
                elif family=='gate_historical':z=ecg.historical_gate(y,p['k'],p['ratio'])
                else:z=ecg.apply(y,f,p);shared=feature_ms if family not in ['raw','lowpass'] else 0
                ms=shared+(time.perf_counter()-st)*1000
                runtimes.append(dict(database=database,record=clip['record'],method=name,duration_s=n/360,ms=ms))
                rows.append(dict(database=database,record=clip['record'],start_s=clip['start'],noise=kind,schedule=schedule,db=db,seed=seed,method=name,**full_metrics(clip,z,y,m)))
                zs[name]=z
            if example is None and kind=='awgn' and schedule=='burst' and db==6:
                example=dict(x=clip['x'],y=y,env=m['env'],**zs)
        print('Evaluated',database,clip['record'],clip['start'],'rows',len(rows),flush=True)
    pd.DataFrame(rows).to_csv(OUT/(database+'_raw.csv'),index=False,encoding='utf-8-sig')
    pd.DataFrame(runtimes).to_csv(OUT/(database+'_timing.csv'),index=False)
    np.savez_compressed(OUT/(database+'_example.npz'),**example)
    dump(OUT/(database+'_run_metadata.json'),dict(seconds=time.perf_counter()-t0,cases=len(rows)//len(conf['best']),rows=len(rows),protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),config_sha256=hashlib.sha256((OUT/'comparison_frozen_config.json').read_bytes()).hexdigest(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pywt=pywt.__version__,wfdb=wfdb.__version__))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','tune','mitdb','ludb']);args=p.parse_args()
    if args.phase=='audit':audit()
    elif args.phase=='tune':dev_tune()
    else:run_set(args.phase)

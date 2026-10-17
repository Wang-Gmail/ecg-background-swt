from run_research import *
v.N=10800;noise=v.load_noise();families=[('native','bior3.1',3),('common','sym4',4)]
params=[(tag,dict(wavelet=w,level=lev,intervals=k,alpha=a,shrink=shrink)) for tag,w,lev in families for k in [2,3,4,5] for a in [.25,.5,1,2,3] for shrink in ['hard','soft']]
scores=np.zeros((len(params),3));n=0
for clip in v.load_clips(CONFIG['development_records']):
    for kind in ['awgn','ma']:
        for db in [0,6,12]:
            for seed in [11,22]:
                y,m=v.corrupt(clip,kind,'burst',db,seed,noise);power=np.mean(clip['x'][m['valid']]**2)
                caches={(w,k):ecg.idt_cache(y,w,lev,k) for tag,w,lev in families for k in [2,3,4,5]}
                for i,(tag,p) in enumerate(params):
                    z=ecg.idt_apply(caches[p['wavelet'],p['intervals']],p['alpha'],p['shrink']);e=(z-clip['x'])**2
                    scores[i]+=[np.sqrt(np.mean(e[m['active']])/power),np.sqrt(np.mean(e[m['quiet']])/power),np.mean(e[m['valid']])/power]
                n+=1
    print('IDT follow-up tuning',clip['record'],clip['start'],flush=True)
scores/=n;cap=json.loads((OUT/'comparison_frozen_config.json').read_text())['active_cap'];selected={};selection={}
for tag,w,lev in families:
    ids=[i for i,(fam,p) in enumerate(params) if fam==tag];feasible=[i for i in ids if scores[i,0]<=cap]
    i=min(feasible,key=lambda i:scores[i,1]) if feasible else min(ids,key=lambda i:scores[i,2])
    selected[tag]=params[i][1];selection[tag]=dict(feasible=bool(feasible),active_nrmse=float(scores[i,0]),quiet_nrmse=float(scores[i,1]),whole_nmse=float(scores[i,2]),candidates=len(ids))
dump(OUT/'idt_followup_config.json',dict(best=selected,selection=selection,note='Additional baseline sensitivity after primary extension inspection; parameters still selected only on original development clips. K=3..5 uses a 25ms candidate grid; K=2 exact one-change search. No proposed-method retuning.'))
pd.DataFrame([dict(family=tag,parameters=json.dumps(p),active_nrmse=s[0],quiet_nrmse=s[1],whole_nmse=s[2]) for (tag,p),s in zip(params,scores)]).to_csv(OUT/'idt_followup_dev_scores.csv',index=False)
print('IDT follow-up selected',selected,selection,flush=True)
rows=[]
for clip in ludb_clips():
    v.N=len(clip['x'])
    for kind in ['awgn','ma']:
        for db in [0,6,12]:
            for seed in [4101,4102,4103]:
                y,m=contaminate(clip,kind,'burst',db,seed,noise)
                for tag,p in selected.items():
                    cache=ecg.idt_cache(y,p['wavelet'],p['level'],p['intervals']);z=ecg.idt_apply(cache,p['alpha'],p['shrink'])
                    rows.append(dict(database='ludb',record=clip['record'],start_s=0,noise=kind,schedule='burst',db=db,seed=seed,method='idt_interval_'+tag,**full_metrics(clip,z,y,m)))
    print('IDT follow-up tested',clip['record'],flush=True)
d=pd.DataFrame(rows);d.to_csv(OUT/'idt_followup_raw.csv',index=False)
d.groupby(['noise','method'])[['quiet_nrmse','active_nrmse','rmse_mv']].mean().to_csv(OUT/'idt_followup_summary.csv')
print(d.groupby('method')[['quiet_nrmse','active_nrmse','rmse_mv']].mean().to_string(),flush=True)

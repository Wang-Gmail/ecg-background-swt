from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1]
extra=Path(os.environ.get('ECG_SCIENCE_LIB',str(ROOT/'work/reprodeps')))
if extra.exists():sys.path.insert(0,str(extra))
os.environ['MPLCONFIGDIR']=str(ROOT/'work'/'mplconfig')
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
METHODS=['raw','lowpass_tuned','local_frozen','local_weak_alpha_2','local_weak_alpha_1','gate_common_tuned','gate_historical_frozen','idt_common_tuned','idt_native_original','background_frozen']
LABELS={'raw':'Input','lowpass_tuned':'Low-pass 45 Hz','local_frozen':'Local SWT (a=3)','local_weak_alpha_2':'Local SWT (a=2)','local_weak_alpha_1':'Local SWT (a=1)','gate_common_tuned':'Common-structure gate','gate_historical_frozen':'Historical gate','idt_common_tuned':'IDT common structure','idt_native_original':'IDT bior3.1','background_frozen':'Background calibration'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':300})

def boot_compare(df,other,metric,noise,db):
    p=df.groupby(['record','method'])[metric].mean().unstack()
    a=p['background_frozen'].to_numpy();b=p[other].to_numpy();keep=np.isfinite(a)&np.isfinite(b);a=a[keep];b=b[keep]
    if not len(a) or np.mean(b)<=1e-20:return None
    rng=np.random.default_rng(8122026);ix=rng.integers(0,len(a),size=(20000,len(a)))
    ratio=100*(np.mean(a[ix],axis=1)/np.mean(b[ix],axis=1)-1)
    ci=np.quantile(ratio,[.025,.975])
    return dict(database=db,noise=noise,comparison=other,metric=metric,records=len(a),change_pct=100*(a.mean()/b.mean()-1),ci_low_pct=ci[0],ci_high_pct=ci[1])

allframes=[];summary=[];comparisons=[];failures=[];recorddata=[];timing=[]
for database in ['mitdb','ludb']:
    p=OUT/(database+'_raw.csv')
    if not p.exists():continue
    d=pd.read_csv(p);allframes.append(d)
    metrics=[c for c in d.columns if c.endswith('_mv') or c.endswith('_nrmse') or c.endswith('_db') or c.endswith('_pct')]
    s=d.groupby(['database','noise','schedule','method'])[metrics].mean().reset_index();summary.append(s)
    primary=d[(d.schedule=='burst')&d.noise.isin(['awgn','ma'])]
    for noise in ['combined','awgn','ma']:
        dd=primary if noise=='combined' else primary[primary.noise==noise]
        for other in METHODS:
            if other=='background_frozen' or other not in set(dd.method):continue
            for metric in ['quiet_rmse_mv','active_rmse_mv','quiet_nrmse','active_nrmse','rmse_mv','quiet_p_full_rmse_mv','quiet_qrs_full_rmse_mv','quiet_t_full_rmse_mv']:
                if metric not in dd:continue
                item=boot_compare(dd,other,metric,noise,database)
                if item:comparisons.append(item)
        recorddata.append(dd.groupby(['record','method'])[metrics].mean().reset_index().assign(noise_group=noise,database=database))
    if database=='ludb':
        failures.append(d[~((d.schedule=='burst')&d.noise.isin(['awgn','ma']))].groupby(['noise','schedule','method'])[metrics].mean().reset_index())
    t=pd.read_csv(OUT/(database+'_timing.csv'))
    for method,dd in t.groupby('method'):
        timing.append(dict(database=database,method=method,median_ms=dd.ms.median(),p95_ms=dd.ms.quantile(.95),duration_s=dd.duration_s.iloc[0]))
pd.concat(summary).to_csv(OUT/'grouped_summary.csv',index=False)
pd.DataFrame(comparisons).to_csv(OUT/'paired_record_bootstrap.csv',index=False)
pd.concat(recorddata).to_csv(OUT/'record_level_metrics.csv',index=False)
pd.DataFrame(timing).to_csv(OUT/'runtime_summary.csv',index=False)
if failures:pd.concat(failures).to_csv(OUT/'failure_controls_summary.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(6.95,3.35))
for ax,database in zip(axes,['mitdb','ludb']):
    frames=[d for d in allframes if d.database.iloc[0]==database]
    if not frames:ax.set_visible(False);continue
    d=frames[0];d=d[(d.schedule=='burst')&d.noise.isin(['awgn','ma'])]
    g=d.groupby('method')[['active_nrmse','quiet_nrmse']].mean()
    markers=['o','s','^','v','<','>','P','X','D','*']
    for method,m in zip(METHODS,markers):
        if method not in g.index:continue
        x,y=g.loc[method];ax.scatter(x,y,s=110 if method=='background_frozen' else 45,marker=m,label=LABELS[method])
    ax.set(xlabel='Active-region normalized RMSE',ylabel='Quiet-region normalized RMSE',title=database.upper()+' | smaller is better');ax.grid(alpha=.18)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7.5,bbox_to_anchor=(.5,-.02));fig.tight_layout(rect=(0,.25,1,1));fig.savefig(FIG/'region_tradeoff.png',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(6.95,3.1))
for ax,database in zip(axes,['mitdb','ludb']):
    frames=[d for d in allframes if d.database.iloc[0]==database]
    if not frames:ax.set_visible(False);continue
    d=frames[0];d=d[(d.schedule=='burst')&d.noise.isin(['awgn','ma'])]
    g=d.groupby(['noise','method']).snr_gain_db.mean().unstack();x=np.arange(2);width=.2
    for i,method in enumerate(['local_frozen','gate_historical_frozen','idt_common_tuned','background_frozen']):
        ax.bar(x+(i-1.5)*width,g.loc[['awgn','ma'],method],width,label=LABELS[method])
    ax.set_xticks(x,['AWGN','Recorded muscle artifact']);ax.set(ylabel='Mean whole-record SNR gain (dB)',title=database.upper());ax.axhline(0,color='black',lw=.7);ax.grid(axis='y',alpha=.15)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8);fig.tight_layout(rect=(0,.14,1,1));fig.savefig(FIG/'noise_specific_gain.png',bbox_inches='tight');plt.close(fig)

cfg=json.loads((ROOT/'sources'/'original'/'quiet_frozen_config.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(figsize=(3.45,2.85))
for family in ['local','background']:
    points=np.array(cfg['scores'][family]);params=cfg['grid'][family];sel=np.array([p['shrink']=='firm' for p in params]);p=points[sel]
    ax.scatter(p[:,0],p[:,1],s=15 if family=='background' else 35,alpha=.55,label=family.title()+' candidates')
for family in ['local','background']:
    p=cfg['chosen_dev_scores'][family];ax.scatter(p[0],p[1],s=100,marker='*',label=family.title()+' selected')
ax.axvline(cfg['active_rmse_constraint'],color='black',ls='--',lw=1,label='Active-error cap')
ax.set(xlabel='Active normalized RMSE',ylabel='Quiet normalized RMSE',title='Development tradeoff',xlim=(.28,.65),ylim=(0,.11));ax.legend(fontsize=6.7,loc='upper right');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIG/'development_frontier.png');plt.close(fig)

if (OUT/'ludb_example.npz').exists():
    z=np.load(OUT/'ludb_example.npz');t=np.arange(len(z['x']))/360
    fig,axes=plt.subplots(2,1,figsize=(6.95,3.4),sharex=True)
    axes[0].plot(t,z['y'],color='.75',lw=.6,label='Injected input');axes[0].plot(t,z['x'],color='black',lw=.9,label='Reference record');axes[0].plot(t,z['background_frozen'],lw=.9,label='Background calibration');axes[0].legend(ncol=3,fontsize=8);axes[0].set_ylabel('ECG (mV)')
    for method in ['local_frozen','gate_historical_frozen','background_frozen']:axes[1].plot(t,z[method]-z['x'],lw=.8,label=LABELS[method])
    axes[1].set(xlabel='Time (s)',ylabel='Recovery error (mV)');axes[1].legend(ncol=3,fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'ludb_prespecified_example.png');plt.close(fig)

decision=[]
for noise in ['combined','awgn','ma']:
    c=pd.DataFrame(comparisons)
    if c.empty:continue
    q=c[(c.database=='ludb')&(c.noise==noise)&(c.comparison=='local_frozen')&(c.metric=='quiet_nrmse')]
    a=c[(c.database=='ludb')&(c.noise==noise)&(c.comparison=='local_frozen')&(c.metric=='active_nrmse')]
    if len(q) and len(a):decision.append(dict(noise=noise,quiet_change_pct=float(q.iloc[0].change_pct),active_change_pct=float(a.iloc[0].change_pct),passes_mean_rule=bool(q.iloc[0].change_pct<=-20 and a.iloc[0].change_pct<=2)))
(OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
print('Extension decision',decision,flush=True)
print('Outputs: grouped summary, record bootstrap, controls, runtime, figures',flush=True)

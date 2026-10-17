from pathlib import Path
import sys,os,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'revision_v2';OUT=BASE/'results'
os.environ['ECG_SCIENCE_LIB']=str(ROOT/'work/reprodeps');os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(ROOT/'work'));sys.path.insert(0,str(ROOT/'work/skill_helpers/statistical-analysis'))
import run_research as rr
from assumption_checks import check_normality,detect_outliers
np,pd=rr.np,rr.pd

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def dump(path,obj):path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(path):return pd.read_csv(path)

h=read(OUT/'holdout_raw.csv');s=read(OUT/'stress_raw.csv')
assert len(h)==1620*16 and len(s)==880*16
assert set(h.record)==set(range(2,180,3))
assert set(h.record).isdisjoint(range(1,179,3))
key=['record','noise','db','seed'];assert h.groupby(key).method.nunique().eq(16).all()
metrics=['quiet_nrmse','active_nrmse','quiet_rmse_mv','active_rmse_mv','rmse_mv','snr_gain_db','transition_rmse_mv','boundary_rmse_mv','quiet_p_full_rmse_mv','quiet_qrs_full_rmse_mv','quiet_t_full_rmse_mv']
assert np.isfinite(h[['quiet_nrmse','active_nrmse','rmse_mv']]).all().all()
parts=[]
for noise in ['awgn','ma1','blgn','pooled_primary']:
    subset=h[h.noise.isin(['awgn','ma1'])] if noise=='pooled_primary' else h[h.noise==noise]
    parts.append(subset.groupby(['record','method'])[metrics].mean().reset_index().assign(noise_group=noise))
r=pd.concat(parts,ignore_index=True);r.to_csv(OUT/'record_metrics.csv',index=False)
summary=r.groupby(['noise_group','method'])[metrics].agg(['mean','std','count'])
summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(OUT/'holdout_summary.csv',index=False)
contrasts=[
('fixed_shape','fixed_sqrt_subband','fixed_linear_subband'),
('fixed_source','fixed_sqrt_subband','fixed_sqrt_global'),
('selected_shape','gate_sqrt_subband','gate_linear_subband'),
('selected_source','gate_sqrt_subband','gate_sqrt_global'),
('original_vs_local','background_original','local_frozen'),
('original_vs_historical','background_original','gate_historical'),
('original_vs_common','background_original','gate_common_old'),
('original_vs_nlm','background_original','nlm_tuned'),
('linear_source','gate_linear_subband','gate_linear_global')]
rows=[];loo=[];diagnostics={}
for noise in ['awgn','ma1','blgn','pooled_primary']:
    subset=r[r.noise_group==noise]
    for label,a_name,b_name in contrasts:
        for metric in metrics:
            p=subset.pivot(index='record',columns='method',values=metric)[[a_name,b_name]].dropna()
            a=p[a_name].to_numpy();b=p[b_name].to_numpy();n=len(a)
            if not n or b.mean()==0:continue
            seed=int(hashlib.sha256(f'15102026/{noise}/{label}/{metric}'.encode()).hexdigest()[:8],16)
            ix=np.random.default_rng(seed).integers(0,n,size=(20000,n))
            ratios=100*(a[ix].mean(1)/b[ix].mean(1)-1)
            differences=(a-b)[ix].mean(1)
            ratio_ci=np.percentile(ratios,[2.5,97.5]);diff_ci=np.percentile(differences,[2.5,97.5])
            change=float(100*(a.mean()/b.mean()-1))
            rows.append(dict(noise=noise,contrast=label,method_a=a_name,method_b=b_name,metric=metric,records=n,mean_a=a.mean(),mean_b=b.mean(),change_pct=change,ci_low_pct=ratio_ci[0],ci_high_pct=ratio_ci[1],paired_difference=float((a-b).mean()),diff_ci_low=diff_ci[0],diff_ci_high=diff_ci[1]))
            if metric in ['quiet_nrmse','active_nrmse']:
                influence=[100*(np.delete(a,k).mean()/np.delete(b,k).mean()-1) for k in range(n)]
                loo.append(dict(noise=noise,contrast=label,metric=metric,records=n,min_pct=min(influence),max_pct=max(influence)))
                if label in ['selected_shape','original_vs_local']:
                    difference=a-b
                    diagnostics[f'{noise}/{label}/{metric}']=dict(record_ids=p.index.tolist(),n=n,missing_records=60-n,normality=check_normality(difference,plot=False),outliers=detect_outliers(difference,plot=False),decision='descriptive paired record bootstrap retained; all valid records retained; no normality-based test switching')
pd.DataFrame(rows).to_csv(OUT/'paired_contrasts.csv',index=False)
pd.DataFrame(loo).to_csv(OUT/'leave_one_out.csv',index=False)
dump(OUT/'distribution_checks.json',diagnostics)
stress_record=s.groupby(['record','noise','nominal_fraction','bursts','method'],dropna=False)[metrics].mean().reset_index()
stress_record.to_csv(OUT/'stress_record_metrics.csv',index=False)
stress_rows=[]
for (noise,q,k,method),g in stress_record.groupby(['noise','nominal_fraction','bursts','method']):
    for metric in ['quiet_nrmse','active_nrmse','snr_gain_db','rmse_mv']:
        a=g[metric].dropna().to_numpy();n=len(a)
        if n:
            seed=int(hashlib.sha256(f'stress/{noise}/{q}/{k}/{method}/{metric}'.encode()).hexdigest()[:8],16)
            ix=np.random.default_rng(seed).integers(0,n,size=(20000,n));ci=np.percentile(a[ix].mean(1),[2.5,97.5])
            point=float(a.mean());lo=float(ci[0]);hi=float(ci[1])
        else:point=lo=hi=np.nan
        stress_rows.append(dict(noise=noise,nominal_fraction=q,bursts=int(k),method=method,metric=metric,records=n,mean=point,ci_low=lo,ci_high=hi))
pd.DataFrame(stress_rows).to_csv(OUT/'stress_summary.csv',index=False)
decisions=[];c=pd.DataFrame(rows)
for noise in ['awgn','ma1','blgn','pooled_primary']:
    for contrast in ['original_vs_local','selected_shape','selected_source','fixed_shape','fixed_source']:
        z=c[(c.noise==noise)&(c.contrast==contrast)]
        q=float(z[z.metric=='quiet_nrmse'].iloc[0].change_pct);a=float(z[z.metric=='active_nrmse'].iloc[0].change_pct)
        decisions.append(dict(noise=noise,contrast=contrast,quiet_change_pct=q,active_change_pct=a,working_rule_pass=bool(q<=-20 and a<=2)))
dump(OUT/'decision_ledger.json',decisions)
frozen=json.loads((OUT/'frozen_config.json').read_text())
dump(OUT/'analysis_integrity.json',dict(status='passed',holdout_inputs=1620,stress_inputs=880,holdout_records=60,stress_subset_records=20,stored_methods=16,clinical_validation=False,analysis='paired record descriptive bootstrap, 20000 repetitions',frozen_selection=frozen['chosen'],protocol_hashes=frozen['source_hashes'],raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'holdout_raw.csv',OUT/'stress_raw.csv']}))
print('NEW HOLDOUT CONTRASTS',flush=True)
print(pd.DataFrame(decisions).to_string(index=False),flush=True)
print('Overall mean errors:')
print(r[r.noise_group=='pooled_primary'].groupby('method')[['quiet_rmse_mv','active_rmse_mv','rmse_mv','snr_gain_db']].mean().to_string())

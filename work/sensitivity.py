exec((__import__('pathlib').Path(__file__).parent/'run_research.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0])
settings=[('reference',1.,.2,3.,2.),('window_0.5',.5,.2,3.,2.),('window_2',2.,.2,3.,2.),('quantile_0.1',1.,.1,3.,2.),('quantile_0.3',1.,.3,3.,2.),('background_c1',1.,.2,3.,1.),('background_c3',1.,.2,3.,3.),('alpha_2',1.,.2,2.,2.),('alpha_4',1.,.2,4.,2.)]
rows=[];v.N=10800;noise=v.load_noise(['ma'])
for clip in v.load_clips(CONFIG['development_records']):
    for kind in ['awgn','ma']:
        for db in [0,6,12]:
            for seed in [11,22]:
                y,m=v.corrupt(clip,kind,'burst',db,seed,noise);scale=np.sqrt(np.mean(clip['x'][m['valid']]**2))
                for name,window,q,alpha,c in settings:
                    z=ecg.apply(y,ecg.features(y,window=window,q=q),dict(family='background',alpha=alpha,c=c,shrink='firm'))
                    e=(z-clip['x'])**2
                    rows.append(dict(record=clip['record'],start=clip['start'],noise=kind,db=db,seed=seed,setting=name,active_nrmse=np.sqrt(np.mean(e[m['active']]))/scale,quiet_nrmse=np.sqrt(np.mean(e[m['quiet']]))/scale))
d=pd.DataFrame(rows);d.to_csv(OUT/'development_sensitivity_raw.csv',index=False)
s=d.groupby('setting')[['active_nrmse','quiet_nrmse']].mean();s.to_csv(OUT/'development_sensitivity_summary.csv');print(s.to_string())

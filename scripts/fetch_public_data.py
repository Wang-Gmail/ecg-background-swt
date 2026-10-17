from pathlib import Path
import shutil,urllib.request,hashlib
import wfdb
ROOT=Path(__file__).resolve().parents[1]
from wfdb.io import download
for db,expected in [('mitdb','1.0.0'),('nstdb','1.0.0'),('ludb','1.0.1')]:
    assert download.get_version(db)==expected, 'Database latest version changed; use the pinned source version documented in README.'
for db,records in [('mitdb',['100','101','103']),('nstdb',['ma','em','bw'])]:
    target=ROOT/'data'/db;target.mkdir(parents=True,exist_ok=True)
    wfdb.dl_database(db,str(target),records=records,annotators='all',keep_subdirs=False,overwrite=False)
target=ROOT/'data/ludb';target.mkdir(parents=True,exist_ok=True)
records=[str(n) for n in range(1,181) if n%3 in (0,2)]
# Official LUDB records are under its data/ directory.
wfdb.dl_database('ludb',str(target),records=['data/'+n for n in records],annotators='all',keep_subdirs=False,overwrite=False)
urllib.request.urlretrieve('https://physionet.org/files/ludb/1.0.1/SHA256SUMS.txt',target/'SHA256SUMS.txt')
official={line.split()[1]:line.split()[0] for line in (target/'SHA256SUMS.txt').read_text().splitlines() if len(line.split())==2}
dest=ROOT/'revision_v2/data/ludb_holdout';dest.mkdir(parents=True,exist_ok=True)
for n in records:
    for ext in ['dat','hea','ii']:
        p=target/(n+'.'+ext)
        if ext in ['dat','hea']:
            expected=official['data/'+p.name]
            assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,p
        if int(n)%3==2:shutil.copy2(p,dest/p.name)
(ROOT/'new_experiments/public_data').mkdir(parents=True,exist_ok=True)
print('Downloaded public ECG/noise data and verified LUDB source checksums.')

from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
for rel,expected in manifest['files'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
frozen=json.loads((ROOT/'revision_v2/results/frozen_config.json').read_text())
assert frozen['chosen']['gate_sqrt_subband']==dict(source='subband',shape='sqrt',alpha=3,c=2)
import numpy as np
from src.ecg_denoise import denoise_ecg
y=np.sin(np.arange(3600)*2*np.pi/360)
z=denoise_ecg(y)
assert z.shape==y.shape and np.isfinite(z).all()
assert np.array_equal(y,np.sin(np.arange(3600)*2*np.pi/360))
print('Release structure, hashes, frozen configuration and observation-only inference checks passed.')

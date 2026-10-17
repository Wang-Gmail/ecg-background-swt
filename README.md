# Background calibrated SWT ECG denoising

Reproducible implementation and frozen settings for studying regional error tradeoffs and generalization limits in ECG denoising. This is an offline, noncausal research implementation. It is not a validated clinical device or diagnostic algorithm.

## Quick inference

Use Python 3.12 and install `python -m pip install -r requirements.txt`. From the repository root:

```python
from src.ecg_denoise import denoise_ecg
recovered = denoise_ecg(observed_ecg_mv, fs=360)
```

The input is a finite one-dimensional ECG in mV at 360 Hz. The function receives only the observation. The default frozen square-root subband setting is a=3, c=2. It uses centered windows and full-clip background quantiles, and therefore requires future samples.

## Settings and saved results

- `revision_v2/results/frozen_config.json`: all original selected settings, selection objective, feasibility and source hashes.
- `new_experiments/development_selection.json` and `record_allocation.csv`: separate twenty-record development, validation and timing groups. These do not replace the original frozen settings.
- `results/revision/emd_frozen.json`: EMD baseline selected on the original 72 MIT-BIH development inputs, before its holdout evaluation.
- `results/revision/`: absolute voltage errors, PRD, per-record fixed-XQRS metrics, paired contrasts and integrity checks.
- `results/original/`: original summary and contrast tables, partitioned into numbered CSV files without dropping rows.

All reported comparison units are ECG records, or participants for SimEMG. Multiple noise seeds are repeated conditions, not independent subjects. Public results include failures: persistent/fragmented interference, selection reversal and minimal SimEMG improvement.

## Data preparation

Raw databases are not redistributed here. Download the public sources with `python scripts/fetch_public_data.py` after checking their database licenses:

- MIT-BIH Arrhythmia Database 1.0.0, records 100, 101 and 103: https://physionet.org/content/mitdb/1.0.0/
- MIT-BIH Noise Stress Test Database 1.0.0, ma, em, bw: https://physionet.org/content/nstdb/1.0.0/
- LUDB 1.0.1, lead II, selected records: https://physionet.org/content/ludb/1.0.1/
- SimEMG V1, DOI https://doi.org/10.17632/yx5pb66hwz.1. Download the official archive and save it as `new_experiments/public_data/simemg-1.zip`; expected SHA256 is recorded in `new_experiments/simemg_metadata.json`.

The downloader stores data at the paths expected by the original code, retrieves LUDB annotations and the official checksum list, and verifies available checksums. Lead and sample-rate transformations are specified in code.

## Recompute original and additional analyses

Run from the repository root, with dependencies installed in the active environment:

```text
python revision_v2/code/run_v2.py verify
python revision_v2/code/run_v2.py holdout
python revision_v2/code/run_v2.py stress
python revision_v2/code/analyze_v2.py
python new_experiments/run_extension.py development
python new_experiments/run_validation.py
python new_experiments/analyze_validation.py
python new_experiments/run_extension.py temporal
python new_experiments/analyze_temporal.py
python new_experiments/run_independent_noise.py
python new_experiments/analyze_independent_noise.py
python new_experiments/run_simemg.py
python new_experiments/analyze_simemg.py
python new_experiments/run_morphology.py
python new_experiments/analyze_morphology.py
python work/reviewer_revision_20261017/run_revision_analysis.py smoke
python work/reviewer_revision_20261017/run_revision_analysis.py select
python work/reviewer_revision_20261017/run_revision_analysis.py evaluate
python work/reviewer_revision_20261017/run_revision_analysis.py summarize
```

The original holdout run generates the per-input file and hashes used by the additional analysis. Full NLM evaluation is computationally expensive. Running original selection is optional: `run_v2.py development` reproduces the original selection and writes settings, so use a fresh copy if preserving the bundled frozen file.

The additional protocol is in `work/reviewer_revision_20261017/ANALYSIS_PROTOCOL.md`. The EMD baseline uses cubic envelopes, six-IMF maximum, 512-sample reflection and firm thresholding of early IMFs. It is a fully specified baseline, not a claimed exact reproduction of a published complete EMD denoiser. The fixed XQRS task uses WFDB 4.3.1 defaults, learn=True, a 150-ms matching window and scoring from 1 to 9 seconds. It does not establish diagnostic benefit.

PRD equals 100 times whole scoring-region RMSE divided by reference RMS after whole-clip mean centering; it is not an independent outcome from normalized RMSE. Regional percentages and voltage-mean ratios use different aggregation and need not coincide.

## Reproduction and provenance

`RELEASE_MANIFEST.json` lists files and SHA256 values. Generated integrity checks verify regenerated input hashes and original-method metric parity. `python scripts/check_release.py` checks package structure, source hashes, frozen settings and the inference interface without downloading data. For precise version provenance cite an immutable Git commit permalink.

See `NOTICE.md` and the original NLM license for attribution. Source data and installed dependencies retain their own terms.

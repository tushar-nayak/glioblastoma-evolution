# Publication Protocol

This document defines the minimum protocol for turning the current reproducibility checkpoint into a publication-ready experiment package.

## Scope

Primary question: can a history-conditioned Neural ODE forecast future multi-modal GBM MRI slices more accurately than a persistence baseline that copies the latest available scan?

Primary comparator: latest-history persistence using the same target slice stack and registration path as the learned model.

Primary metric: mean squared error averaged across FLAIR, T1, T2, and CT1 target slices.

Secondary metrics:

- Mean absolute error.
- Per-modality MSE and MAE.
- Relative FLAIR volume difference from thresholded FLAIR slices.
- Positive versus nonpositive patient-level relative improvement.

## Dataset Requirements

- Use only patients with at least three usable longitudinal scans when `--holdout-last-pair` is enabled.
- Each included week must contain all four modalities: FLAIR, T1, T2, and CT1.
- Record the exact LUMIERE dataset source, access date, data-use terms, and any exclusion criteria.
- Keep raw NIfTI data outside git. Commit only derived summary tables and small manuscript assets.

## Preprocessing

Run registration preprocessing before final training:

```bash
python3 scripts/preprocess_lumiere_registration.py \
  --data-dir path/to/LUMIERE/Imaging \
  --output-dir data/lumiere_registered
```

The generated `registration_manifest.json` must be archived with the experiment outputs. It records whether each patient/week/target pair was generated or reused from cache.

## Final Training Run

Use a fixed run name, fixed seed, explicit device, and pre-registered data:

```bash
python3 scripts/run_neural_ode_pipeline.py \
  --lumiere \
  --data-dir path/to/LUMIERE/Imaging \
  --registered-data-dir data/lumiere_registered \
  --separate-patient-runs \
  --holdout-last-pair \
  --epochs 40 \
  --batch-size 1 \
  --model-size standard \
  --device cuda \
  --seed 7 \
  --run-name lumiere_publication_v1
```

If CUDA is unavailable, use `--device cpu` or `--device mps` and record the reason. Do not mix devices within a frozen result table unless the manuscript explicitly reports that difference.

## Required Outputs

For each run directory, retain:

- `run_summary.json`
- `train_history_metrics.csv`
- `holdout_history_metrics.csv`
- `all_history_metrics.csv`
- `baseline_train_history_metrics.csv`
- `baseline_holdout_history_metrics.csv`
- `baseline_all_history_metrics.csv`
- Prediction figures for held-out targets.
- Model checkpoint, stored outside git unless the artifact is small enough and approved for release.

For repository-tracked results, regenerate aggregate tables:

```bash
python3 scripts/summarize_run_metrics.py \
  --runs-dir runs \
  --pattern 'lumiere_publication_v1_Patient-*/run_summary.json' \
  --output-dir results \
  --prefix lumiere_publication_v1_metrics \
  --split holdout
```

Final publication claims should use `--split holdout`. All rows in the resulting table should have `metric_split=holdout`.

## Acceptance Criteria

- Unit tests pass with `python3 -m unittest discover -s tests`.
- Scripts compile with `python3 -m py_compile scripts/*.py`.
- The aggregation table includes every eligible patient exactly once.
- Every publication table reports both model and persistence-baseline metrics.
- The manuscript reports the number of eligible, excluded, positive-improvement, and nonpositive-improvement patients.
- Figures are regenerated from the same frozen run prefix as the tables.
- Claims are limited to the frozen holdout table unless additional external validation is added.

## Known Limitations To Report

- The current model predicts 2D slice stacks rather than full 3D volumes.
- The persistence baseline is strong for slowly changing scans and must remain central in interpretation.
- Registration quality can affect all downstream metrics.
- Final clinical claims require a locked cohort definition, data-use language, and independent holdout reporting.

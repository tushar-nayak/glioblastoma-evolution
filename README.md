# glioblastoma-evolution

This branch, `neural-ode-implementation`, is focused on a cleaned implementation of the recovered **attention U-Net + Neural ODE** approach for longitudinal glioblastoma MRI prediction.

The main script for this branch is:

- [`scripts/run_neural_ode_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_neural_ode_pipeline.py)

The patient data used by this branch is:

- [`patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_007)
- [`patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_067)

The original recovered Neural ODE source that this implementation is based on is primarily:

- [`recovered_files/9b9726a89f5efdfb95654ce40c8925ccbe40cfbb.py`](/Users/tushar/Documents/Repositories/Glioblastoma/recovered_files/9b9726a89f5efdfb95654ce40c8925ccbe40cfbb.py)

## What This Branch Implements

The recovered code was a rough notebook/script hybrid. It contained:

- a 2D attention U-Net
- a feature-space ODE block using `torchdiffeq.odeint`
- temporal conditioning with scan weeks
- ad hoc training and plotting logic mixed into one file

This branch extracts that idea into a clean runnable pipeline that:

1. loads repo-local patient data instead of hardcoded external paths
2. builds longitudinal prediction pairs from scan weeks
3. uses multiple context weeks as input
4. predicts the future **center slice** for **all four MRI modalities**
5. supports separate-per-patient runs
6. supports a simple holdout split
7. compares the learned model against a persistence baseline

## Model Summary

The Neural ODE pipeline is a **2D slice-based forecasting model**, not a full 3D volumetric model.

At a high level:

1. For each context week, the script takes a few slices around the center slice.
2. Those slices from all four modalities are flattened into a multi-channel 2D tensor.
3. An attention U-Net maps that tensor into latent features.
4. A Neural ODE evolves those latent features over a time interval `dt`.
5. A decoder maps the evolved features to the predicted future center slice.

The model predicts:

- `FLAIR`
- `T1`
- `T2`
- `CT1`

for the target week.

## Data Construction

The current implementation uses:

- 3 context weeks by default
- center-slice neighborhood offsets `[-1, 0, 1]`

So for each week:

- 4 modalities
- 3 nearby slices per modality

This gives:

```text
4 modalities x 3 slices = 12 input channels per context week
```

With 3 context weeks, the context tensor becomes:

```text
3 weeks x 12 channels/week = 36 channels
```

The target is the center slice at the future week with 4 channels:

```text
[FLAIR, T1, T2, CT1]
```

## Current Patient Timelines

The usable weeks in this repo are:

- `patient_007`: `55, 64, 75, 89, 105`
- `patient_067`: `92, 109, 124, 136, 152`

With `context_size=3`, each patient yields only 3 forecasting pairs:

- context `[w1, w2, w3] -> target w4`
- context `[w1, w2, w3] -> target w5`
- context `[w2, w3, w4] -> target w5`

That means this branch is operating in a very small-data regime.

## Setup

Create a local environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

This branch requires:

- `torch`
- `torchdiffeq`
- `nibabel`
- `matplotlib`
- `numpy`

## Main Run Commands

### Separate per-patient Neural ODE run

This is the most appropriate mode for the current dataset constraint:

```bash
.venv/bin/python scripts/run_neural_ode_pipeline.py \
  --epochs 20 \
  --lr 3e-4 \
  --holdout-last-pair \
  --separate-patient-runs \
  --model-size tiny \
  --run-name neural_ode_smoke
```

### Short all-four-modality smoke test

```bash
.venv/bin/python scripts/run_neural_ode_pipeline.py \
  --epochs 5 \
  --lr 3e-4 \
  --holdout-last-pair \
  --separate-patient-runs \
  --model-size tiny \
  --run-name neural_ode_all4_smoke
```

### Strict future-target holdout

The current `--holdout-last-pair` behavior on this branch is intentionally strict:

- for each patient, any sample whose `target_week` is the latest week is excluded from training
- for `patient_067`, that means `week 152` is fully held out from training
- for `patient_007`, that means `week 105` is fully held out from training

This is the correct mode if you want to simulate future prediction rather than leak the final timepoint into training.

### Recommended Neural ODE run on this branch

If the goal is to see the most usable generated images from this Neural ODE approach while preserving strict target holdout, the current best run setting is:

```bash
.venv/bin/python scripts/run_neural_ode_pipeline.py \
  --epochs 30 \
  --lr 1e-4 \
  --context-size 2 \
  --holdout-last-pair \
  --separate-patient-runs \
  --model-size standard \
  --run-name neural_ode_context2_strict_holdout_30
```

This gives each patient more trainable pairs than `context_size=3` while still keeping the latest target week fully out of training.

### Strict-holdout results

The best run on this branch so far is the strict-holdout `context_size=2` setup above. The table below reports holdout performance against the persistence baseline.

| Patient | Held-out future week | Train pairs | Holdout pairs | Neural ODE MSE | Baseline MSE | Neural ODE MAE | Baseline MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `patient_007` | `105` | 3 | 3 | 0.04119 | 0.00399 | 0.16927 | 0.03153 |
| `patient_067` | `152` | 3 | 3 | 0.02426 | 0.00752 | 0.14075 | 0.04469 |

FLAIR volume behavior on holdout was mixed:

- `patient_007`: Neural ODE relative FLAIR volume difference `20.6351`, baseline `0.7743`
- `patient_067`: Neural ODE relative FLAIR volume difference `0.3794`, baseline `0.7029`

For reference, the stricter `context_size=3` run was worse overall:

| Patient | Holdout pairs | `context_size=3` Neural ODE MSE | `context_size=3` Baseline MSE | `context_size=3` Neural ODE MAE | `context_size=3` Baseline MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `patient_007` | 2 | 0.03812 | 0.00392 | 0.17049 | 0.03168 |
| `patient_067` | 2 | 0.03821 | 0.00518 | 0.17012 | 0.03472 |

So the current conclusion on this branch is precise:

- `context_size=2` is the better Neural ODE configuration for this limited dataset
- the implementation is functional and produces images
- the learned model still does not beat the simple persistence baseline under strict future holdout

### Shared run across both patients

```bash
.venv/bin/python scripts/run_neural_ode_pipeline.py \
  --epochs 20 \
  --lr 3e-4 \
  --holdout-last-pair \
  --model-size tiny \
  --run-name neural_ode_shared
```

## Important Arguments

The most relevant options are:

- `--patients patient_007 patient_067`
  Restrict to specific patients.

- `--context-size 3`
  Number of context weeks per sample.

- `--slice-offsets -1 0 1`
  Relative slice offsets around the center slice.

- `--holdout-last-pair`
  Removes the latest forecasting pair for each patient from training and reports holdout metrics on it.

- `--separate-patient-runs`
  Trains one independent Neural ODE model per patient instead of a shared model.

- `--model-size tiny|standard`
  Chooses a smaller or larger attention U-Net/ODE configuration.

## Outputs

Each run writes a folder under [`runs/`](/Users/tushar/Documents/Repositories/Glioblastoma/runs).

Typical contents:

- `attention_unet_neural_ode_<size>.pt`
- `loss_curve.png`
- `run_summary.json`
- per-patient prediction images

Examples from this branch:

- [`runs/neural_ode_smoke_patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_smoke_patient_007)
- [`runs/neural_ode_smoke_patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_smoke_patient_067)
- [`runs/neural_ode_all4_smoke_patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_all4_smoke_patient_007)
- [`runs/neural_ode_all4_smoke_patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_all4_smoke_patient_067)
- [`runs/neural_ode_full_strict_holdout_30_patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_full_strict_holdout_30_patient_007)
- [`runs/neural_ode_full_strict_holdout_30_patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_full_strict_holdout_30_patient_067)
- [`runs/neural_ode_context2_strict_holdout_30_patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_context2_strict_holdout_30_patient_007)
- [`runs/neural_ode_context2_strict_holdout_30_patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/neural_ode_context2_strict_holdout_30_patient_067)

The summary JSON includes:

- train metrics
- holdout metrics
- baseline metrics
- per-modality MSE for all four modalities
- prediction visualization paths

## Baseline

This branch includes a simple persistence baseline:

```text
predict the target slice as the latest context week center slice
```

This is important because with small longitudinal datasets, a persistence baseline can be very strong.

## Current Status

The implementation is working, but the current Neural ODE results are weak.

On the existing smoke runs, the learned Neural ODE underperforms the simple persistence baseline on holdout data. That is true even when evaluating all four modalities explicitly.

The `context_size=2` strict-holdout run improves the generated outputs compared with the stricter `context_size=3` setup because it leaves more training pairs per patient, but it still does not beat the baseline.

This means:

- the recovered approach has now been implemented cleanly
- the training and inference path is functional
- but the present dataset is too small for this model family to perform convincingly

## Current Limitations

This branch has several important limitations:

1. It predicts only 2D center slices, not full 3D volumes.
2. The number of longitudinal examples is extremely small.
3. Holdout evaluation is based on very few samples.
4. The baseline is hard to beat because adjacent scans are still similar.
5. The model has substantially more flexibility than the amount of data can really support.

## Relationship To The Physics Branch

This branch does not replace the physics-informed model. It is a separate implementation path based on the recovered Neural ODE idea.

The physics script is still present in the repo:

- [`scripts/run_physics_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_physics_pipeline.py)

But this branch is specifically for the Neural ODE implementation and evaluation.

## Recommended Interpretation

Treat this branch as:

- a cleaned reproduction of the recovered Neural ODE experiment
- a runnable reference implementation
- a comparison point against the physics-informed approach

Do not treat it as a validated predictive model. With the data available in this repository, it is an exploratory implementation only.

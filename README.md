# glioblastoma-evolution

This branch, `history-conditioned-forecast`, turns the recovered Neural ODE idea into a **prefix-history forecasting** experiment.

The goal is:

```text
predict x_tn from {x_t0, x_t1, ..., x_t(n-1)}
```

where each `x_t` is a four-modality MRI observation and `tn` is a future scan we want to forecast from all earlier scans.

The main script on this branch is:

- [`scripts/run_neural_ode_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_neural_ode_pipeline.py)

The patient data used by this branch is:

- [`patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_007)
- [`patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_067)

The recovered source this branch is based on is:

- [`recovered_files/9b9726a89f5efdfb95654ce40c8925ccbe40cfbb.py`](/Users/tushar/Documents/Repositories/Glioblastoma/recovered_files/9b9726a89f5efdfb95654ce40c8925ccbe40cfbb.py)

## What This Branch Implements

The recovered code was a rough notebook/script hybrid. It contained:

- a 2D attention U-Net
- a feature-space ODE block using `torchdiffeq.odeint`
- temporal conditioning with scan weeks
- ad hoc training and plotting logic in one file

This branch turns that into a runnable pipeline that:

1. loads repo-local patient data
2. builds one sample per target week from the full earlier scan history
3. predicts the future center slice for all four modalities
4. keeps the latest target week fully out of training when holdout is enabled
5. compares the learned model against a persistence baseline

## Model Summary

The model is still a 2D slice-based Neural ODE, not a 3D volumetric simulator.

At a high level:

1. Each historical week is encoded separately with an attention U-Net.
2. A learned week embedding is added to each encoded week.
3. The encoded history is aggregated into one latent state.
4. A Neural ODE evolves that latent state forward by `dt`.
5. A decoder maps the evolved latent state to the predicted future center slice.

The model predicts all four modalities:

- `FLAIR`
- `T1`
- `T2`
- `CT1`

## Objective And Math

Let each historical week be `x_i`, with `i = 1 ... m`.

Each week contains 4 modalities and a small slice neighborhood, so after flattening the per-week input, the model sees:

```text
4 modalities x 3 slices = 12 channels per week
```

The branch encodes each week independently:

```text
h_i = U(x_i)
```

where `U` is the attention U-Net backbone.

The week index is embedded and added:

```text
\tilde{h}_i = h_i + e(w_i)
```

and the history is aggregated with a masked mean:

```text
z_0 = (sum_i m_i \tilde{h}_i) / (sum_i m_i)
```

where `m_i` is a binary mask for valid history entries.

Then the Neural ODE evolves the latent state:

```math
\frac{dz}{dt} = f_\theta(z, t)
```

and the decoder produces the forecast:

```text
\hat{x}_{tn} = D(z(t_n))
```

The training objective is the usual reconstruction loss on the target future scan:

```text
loss = MSE(prediction, target) + 0.1 * L1(prediction, target)
```

## Data Construction

The usable weeks in this repo are:

- `patient_007`: `55, 64, 75, 89, 105`
- `patient_067`: `92, 109, 124, 136, 152`

In prefix-history mode, each later week becomes a target:

- `55 -> 64`
- `55, 64 -> 75`
- `55, 64, 75 -> 89`
- `55, 64, 75, 89 -> 105`

and similarly for `patient_067`.

With holdout enabled, the latest target week is held out for each patient:

- `patient_007`: `week 105`
- `patient_067`: `week 152`

That leaves 3 training samples and 1 holdout sample per patient.

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

### Prefix-history smoke run

This is the current branch default and the best match for the new objective:

```bash
.venv/bin/python scripts/run_neural_ode_pipeline.py \
  --history-mode prefix \
  --epochs 10 \
  --lr 3e-4 \
  --holdout-last-pair \
  --separate-patient-runs \
  --model-size tiny \
  --run-name history_prefix_smoke
```

### Sliding-window fallback

The script still supports the older sliding-window mode if you want to compare against the prefix-history setup:

```bash
.venv/bin/python scripts/run_neural_ode_pipeline.py \
  --history-mode sliding \
  --context-size 3 \
  --epochs 10 \
  --lr 3e-4 \
  --holdout-last-pair \
  --separate-patient-runs \
  --model-size tiny \
  --run-name history_sliding_smoke
```

## Holdout Rules

The `--holdout-last-pair` flag now means:

- hold out the latest target week for each patient
- do not train on the final future sample
- evaluate the model on that final prefix-to-future forecast

That is the correct setup if the goal is genuine future prediction rather than leaking the final scan into training.

## Smoke-Run Results

The current prefix-history smoke run completed successfully, but it still does not beat the persistence baseline on this very small dataset.

| Patient | Held-out future week | Train samples | Holdout samples | Neural ODE MSE | Baseline MSE | Neural ODE MAE | Baseline MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `patient_007` | `105` | 3 | 1 | 0.04039 | 0.00396 | 0.16903 | 0.03225 |
| `patient_067` | `152` | 3 | 1 | 0.06532 | 0.00479 | 0.22187 | 0.03243 |

Holdout FLAIR volume behavior was also weaker than the baseline:

- `patient_007`: Neural ODE relative FLAIR volume difference `1.0000`, baseline `0.6316`
- `patient_067`: Neural ODE relative FLAIR volume difference `0.9993`, baseline `0.1648`

The takeaway is simple:

- the new prefix-history objective is implemented correctly
- the model trains and produces images
- the dataset is still too small for the Neural ODE to outperform a persistence baseline

## Outputs

Each run writes a folder under [`runs/`](/Users/tushar/Documents/Repositories/Glioblastoma/runs).

Typical contents:

- `attention_unet_neural_ode_<size>.pt`
- `loss_curve.png`
- `run_summary.json`
- per-patient prediction images

Examples from this branch:

- [`runs/history_prefix_smoke_patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/history_prefix_smoke_patient_007)
- [`runs/history_prefix_smoke_patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/history_prefix_smoke_patient_067)

The summary JSON includes:

- train metrics
- holdout metrics
- baseline metrics
- per-modality MSE for all four modalities
- prediction visualization paths

## Important Arguments

The most relevant options are:

- `--history-mode prefix|sliding`
  Chooses full prefix history or the older fixed-window setup.

- `--context-size N`
  Only used when `--history-mode sliding`.

- `--slice-offsets -1 0 1`
  Relative slice offsets around the center slice.

- `--holdout-last-pair`
  Holds out the latest target week for each patient.

- `--separate-patient-runs`
  Trains one independent Neural ODE model per patient instead of a shared model.

- `--model-size tiny|standard`
  Chooses a smaller or larger attention U-Net/ODE configuration.

## Current Status

This branch is the sequence-to-one forecast experiment.

It is working end to end, but on the current dataset the learned Neural ODE still underperforms the persistence baseline. That makes this useful as a controlled experiment, not as a convincing predictor.

## Relationship To The Physics Branch

This branch is separate from the physics-informed pipeline.

The physics script is still present in the repo:

- [`scripts/run_physics_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_physics_pipeline.py)

That branch is about explicit reaction-diffusion dynamics. This branch is about learning a history-conditioned latent forecast from prior scans.

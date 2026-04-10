# glioblastoma-evolution

This branch is the merged, cleaned prefix-history forecast experiment.

It keeps the history-conditioned Neural ODE work and strips out the unrelated physics/recovered-artifact branch baggage.

The goal is:

```text
predict the next MRI state from all earlier MRI states
```

The main script is:

- [`scripts/run_neural_ode_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_neural_ode_pipeline.py)

## Branch Goal

Use the full scan prefix for each patient to predict the next future scan.

For a patient with weeks:

```text
t0, t1, t2, ..., tn
```

the model learns samples like:

```text
{t0} -> t1
{t0, t1} -> t2
{t0, t1, t2} -> t3
...
{t0, ..., t(n-1)} -> tn
```

This is a sequence-to-one forecasting setup rather than a fixed sliding window setup.

## Model

The pipeline is a 2D slice-based Neural ODE.

For each historical week:

1. The four MRI modalities are loaded.
2. A small neighborhood of center slices is flattened into a 2D tensor.
3. An attention U-Net encodes that week.
4. A learned week embedding is added.
5. The encoded history is averaged into one latent state.
6. A Neural ODE evolves that state forward by the time gap `dt`.
7. A decoder predicts the future center slice for all four modalities.

The model outputs:

- `FLAIR`
- `T1`
- `T2`
- `CT1`

## Objective

Let `x_i` be the encoded representation for each observed week and `e(w_i)` the learned week embedding.

The branch computes:

```text
z_0 = mean_i (x_i + e(w_i))
```

with padding masked out when a history prefix is shorter than the maximum batch length.

Then the latent state is evolved with a Neural ODE:

```math
\frac{dz}{dt} = f_\theta(z, t)
```

and the forecast is decoded as:

```text
\hat{x}_{tn} = D(z(t_n))
```

Training minimizes:

```text
MSE(prediction, target) + 0.1 * L1(prediction, target)
```

## Data

The current local dataset includes:

- `patient_007`
- `patient_067`

The branch supports:

- `--history-mode prefix` for full scan-history forecasting
- `--history-mode sliding` for the older fixed-window comparison

## Holdout

The holdout rule is strict:

- the latest target week for each patient is held out
- that future week is never used for training

This simulates genuine future prediction.

## Smoke Results

The current prefix-history smoke run is functional, but the learned model still loses to the persistence baseline on this tiny dataset.

| Patient | Held-out week | Train samples | Holdout samples | Neural ODE MSE | Baseline MSE | Neural ODE MAE | Baseline MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `patient_007` | `105` | 3 | 1 | 0.04039 | 0.00396 | 0.16903 | 0.03225 |
| `patient_067` | `152` | 3 | 1 | 0.06532 | 0.00479 | 0.22187 | 0.03243 |

## Files Kept In This Branch

Only the files needed for this experiment are kept here:

- [`.gitignore`](/Users/tushar/Documents/Repositories/Glioblastoma/.gitignore)
- [`README.md`](/Users/tushar/Documents/Repositories/Glioblastoma/README.md)
- [`requirements.txt`](/Users/tushar/Documents/Repositories/Glioblastoma/requirements.txt)
- [`scripts/run_neural_ode_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_neural_ode_pipeline.py)


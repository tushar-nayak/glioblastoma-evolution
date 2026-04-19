# glioblastoma-evolution

This repository now consolidates the findings from the recovered branches into one mainline summary:

- the physics-informed 3D forecasting branch
- the cleaned Neural ODE implementation
- the prefix-history Neural ODE refinement
- the current slim merge branch that keeps the runnable pipeline and results

The practical outcome is consistent across branches:

- the forecasting pipelines run end to end on the available local data
- the learned Neural ODE variants do not beat a simple persistence baseline on the tiny two-patient cohort
- the prefix-history formulation is the clearest formulation of the forecasting task
- the full LUMIERE cohort is the right next dataset if the goal is a real generalization study

## Main Code

The runnable pipeline in this repository is:

- [`scripts/run_neural_ode_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_neural_ode_pipeline.py)

It supports:

- `--history-mode prefix` for full scan-history forecasting
- `--history-mode sliding` for the older fixed-window comparison
- `--holdout-last-pair` for strict future-target evaluation
- `--separate-patient-runs` for one model per patient

## Data

The local working set currently contains:

- `patient_007`
- `patient_067`

These are the only patients used in the recorded runs in this repo.

The public LUMIERE dataset is much larger:

- 91 GBM patients
- 638 studies
- 2487 MRI images
- 30.33 GB archive

The full archive is the right source for a stronger experiment, but in this session the Figshare download was blocked by a WAF challenge, so the repository still only reflects the small local subset.

## Unified Findings

### Physics-Informed Branch

The physics-informed branch was a separate 3D forecasting attempt.

What it established:

- the code path was runnable on the two local patients
- it served as a feasible baseline for physics-guided dynamics
- it remained exploratory rather than a strong predictive result

This is the main non-Neural-ODE line of work in the repository history. It is not kept as active code in the slim main branch, but its finding still matters:

- the physics-inspired setup was viable
- it did not produce evidence that it beats the simpler persistence baseline
- it is best treated as a historical comparison point, not the current mainline implementation

## Branch Timeline

The work in this repository evolved in four steps:

1. Initial physics branch
   - a 3D physics-informed forecasting attempt
   - runnable on the local patients
   - useful as a historical baseline, but not a baseline winner

2. Cleaned Neural ODE branch
   - the recovered Neural ODE idea was made reproducible
   - strict future-week holdout was added
   - results were compared against persistence

3. Prefix-history refinement
   - the sequence-to-one prefix formulation was made explicit
   - the training/evaluation pipeline stayed functional
   - the learned model still lagged behind the baseline

4. Slim main branch
   - the runnable Neural ODE pipeline was kept
   - the recorded results were preserved
   - the GitHub Pages summary was added

### Neural ODE Branch

The cleaned Neural ODE implementation made the history-conditioning explicit and added strict holdout evaluation.

The main result:

- `context_size=2` was better than `context_size=3` for the limited data
- the model still underperformed the persistence baseline

Strict-holdout summary from the branch:

| Patient | Held-out week | Neural ODE MSE | Baseline MSE | Neural ODE MAE | Baseline MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `patient_007` | `105` | `0.04119` | `0.00399` | `0.16927` | `0.03153` |
| `patient_067` | `152` | `0.02426` | `0.00752` | `0.14075` | `0.04469` |

### Prefix-History Branch

The prefix-history branch is the clearest expression of the forecasting objective:

```text
t0 -> t1
t0, t1 -> t2
t0, t1, t2 -> t3
```

Smoke-run summary:

| Patient | Held-out week | Neural ODE MSE | Baseline MSE | Neural ODE MAE | Baseline MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `patient_007` | `105` | `0.04039` | `0.00396` | `0.16903` | `0.03225` |
| `patient_067` | `152` | `0.06532` | `0.00479` | `0.22187` | `0.03243` |

The conclusion is the same:

- the pipeline is functioning
- the model generates predictions
- the persistence baseline is still much stronger on this dataset size

### Other Non-Neural ODE Approaches

There is one other non-Neural-ODE reference point worth keeping in view:

- the persistence baseline

That baseline is intentionally simple:

```text
predict the next scan as the most recent observed scan
```

Across the recorded experiments it remains the strongest result on the tiny local cohort. That is why the current repository conclusions are framed as feasibility results rather than model wins.

## Website

A simple GitHub Pages site lives in [`docs/`](/Users/tushar/Documents/Repositories/Glioblastoma/docs).

It summarizes:

- the branch findings
- the patient-level results
- the dataset status
- the non-Neural-ODE reference point
- the recommended next step

The GitHub Actions workflow for Pages is in:

- [`.github/workflows/pages.yml`](/Users/tushar/Documents/Repositories/Glioblastoma/.github/workflows/pages.yml)

## Reproducibility

Create an environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Run the current pipeline:

```bash
.venv/bin/python scripts/run_neural_ode_pipeline.py \
  --history-mode prefix \
  --holdout-last-pair \
  --separate-patient-runs \
  --model-size tiny \
  --run-name history_prefix_smoke
```

Outputs are written under [`runs/`](/Users/tushar/Documents/Repositories/Glioblastoma/runs).

## Interpretation

The repo is now best read as a compact record of three things:

1. the recovered forecasting ideas,
2. the implementation and evaluation path,
3. the evidence that the current model family is not yet strong enough on the tiny local cohort.

The next meaningful experiment is to rerun the same pipeline on the full LUMIERE cohort once the archive is available locally.

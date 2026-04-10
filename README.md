# glioblastoma-evolution

This repository contains a small 3D physics-informed glioblastoma forecasting pipeline built around the MRI data in:

- [`patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_007)
- [`patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_067)

The main runnable script is:

- [`scripts/run_physics_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_physics_pipeline.py)

The detailed explanation of the pipeline and math is here:

- [`PIPELINE_EXPLAINED.md`](/Users/tushar/Documents/Repositories/Glioblastoma/PIPELINE_EXPLAINED.md)

## Setup

Create a local environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Common Runs

Train one shared model across both patients:

```bash
.venv/bin/python scripts/run_physics_pipeline.py \
  --epochs 30
```

Train one shared model with a stricter holdout setup:

```bash
.venv/bin/python scripts/run_physics_pipeline.py \
  --epochs 60 \
  --lr 3e-4 \
  --resize-dim 64 64 64 \
  --holdout-last-pair \
  --run-name physics_run_tight_holdout_64
```

Train separate models for each patient:

```bash
.venv/bin/python scripts/run_physics_pipeline.py \
  --epochs 60 \
  --lr 3e-4 \
  --resize-dim 64 64 64 \
  --holdout-last-pair \
  --separate-patient-runs \
  --run-name physics_run_separate_holdout_64
```

Train separate tiny models and compare against the persistence baseline:

```bash
.venv/bin/python scripts/run_physics_pipeline.py \
  --epochs 60 \
  --lr 3e-4 \
  --resize-dim 64 64 64 \
  --holdout-last-pair \
  --separate-patient-runs \
  --model-size tiny \
  --run-name physics_run_tiny_baseline_holdout_64
```

## Outputs

Each run writes to a folder under [`runs/`](/Users/tushar/Documents/Repositories/Glioblastoma/runs) and typically includes:

- a model checkpoint
- `loss_curve.png`
- `forecast.png`
- `therapy.png`
- `run_summary.json`

## Notes

- The current pipeline treats normalized `FLAIR` as a proxy for tumor concentration.
- The model is exploratory and the dataset is very small.
- Results are useful for feasibility experiments, not strong predictive claims.

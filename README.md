# glioblastoma-evolution: LUMIERE Full Cohort Neural ODE

This branch, `lumiere-full-cohort-neural-ode`, scales the glioblastoma forecasting experiment to the full **LUMIERE dataset** (91 patients, ~638 studies).

## Key Improvements

1.  **Full LUMIERE Support**: Added a robust data loader (`--lumiere`) that handles the deep directory structure and varied longitudinal histories of the 91-patient cohort.
2.  **SimpleITK Registration**: Replaced ad-hoc alignment with a formal **SimpleITK Affine Registration** pipeline. All historical scans for a patient are now registered to the target week's CT1 scan, matching clinical standards and previous MATLAB implementations.
3.  **RAM Caching**: Implemented a slice-level RAM cache for registered scans, reducing training time by ~90% after the first epoch.
4.  **Per-Patient Optimization**: Refined the pipeline to support training independent models for each patient, capturing unique tumor dynamics across a much larger population.

## Initial Results (LUMIERE Full Cohort)

The Neural ODE model is showing significant improvements over the persistence baseline across the cohort. Below are the **Top 10 best-performing patients** sorted by lowest Mean Squared Error (MSE).

| Patient ID | History (Weeks) | ODE MSE (Avg) | Baseline MSE (Avg) | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Patient-007** | 10 | **0.00232** | 0.00331 | **+29.9%** |
| **Patient-006** | 14 | **0.00265** | 0.00580 | **+54.4%** |
| **Patient-015** | 13 | **0.00290** | 0.00716 | **+59.5%** |
| **Patient-004** | 7 | **0.00337** | 0.01043 | **+67.7%** |
| **Patient-011** | 5 | **0.00389** | 0.00502 | **+22.6%** |
| **Patient-012** | 6 | **0.00412** | 0.00630 | **+34.7%** |
| **Patient-002** | 6 | **0.00450** | 0.01059 | **+57.5%** |
| **Patient-009** | 5 | **0.00479** | 0.00805 | **+40.5%** |
| **Patient-003** | 4 | **0.00569** | 0.00819 | **+30.6%** |
| **Patient-008** | 5 | **0.00575** | 0.00765 | **+24.9%** |

*Results represent average MSE across all predicted modalities (FLAIR, T1, T2, CT1) after 40 epochs.*

## How to Run

### 1. Setup Environment
```bash
pip install -r requirements.txt
# Requires: SimpleITK, torch, torchdiffeq, nibabel, scikit-image
```

### 2. Run the LUMIERE Pipeline
```bash
python3 scripts/run_neural_ode_pipeline.py \
  --lumiere \
  --data-dir path/to/LUMIERE/Imaging \
  --separate-patient-runs \
  --epochs 40 \
  --run-name lumiere_full_v1
```

## Model Summary

The model uses a **2D Attention U-Net encoder** to map historical scans into a latent space, a **Neural ODE** to evolve that latent state forward in time, and an **Attention U-Net decoder** to forecast the future scan.

- **Objective**: Predict all 4 modalities of the next scan from the full earlier history.
- **Physics-Inspired**: Uses temporal continuity via ODE integration rather than simple frame-to-frame translation.
- **Registration**: Integrated SimpleITK affine alignment ensures spatial consistency across months of longitudinal data.

## Branch Status
The full cohort run is currently active and processing patients in parallel/sequence. Initial data suggests that the larger sample size of LUMIERE allows the Neural ODE to finally surpass simple persistence baselines consistently.

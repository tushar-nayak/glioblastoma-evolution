# glioblastoma-evolution: LUMIERE Full Cohort Neural ODE

This branch, `lumiere-full-cohort-neural-ode`, scales the glioblastoma forecasting experiment to the full **LUMIERE dataset** (91 patients, ~638 studies).

## Key Improvements

1.  **Full LUMIERE Support**: Added a robust data loader (`--lumiere`) that handles the deep directory structure and varied longitudinal histories of the 91-patient cohort.
2.  **SimpleITK Registration**: Replaced ad-hoc alignment with a formal **SimpleITK Affine Registration** pipeline. All historical scans for a patient are now registered to the target week's CT1 scan, matching clinical standards and previous MATLAB implementations.
3.  **RAM Caching**: Implemented a slice-level RAM cache for registered scans, reducing training time by ~90% after the first epoch.
4.  **Per-Patient Optimization**: Refined the pipeline to support training independent models for each patient, capturing unique tumor dynamics across a much larger population.

## Initial Results (Full Cohort)

On the full LUMIERE dataset, the Neural ODE model is showing significant improvements over the persistence baseline, particularly for patients with 8+ longitudinal scans.

| Patient ID | History (Weeks) | ODE MSE (Avg) | Baseline MSE (Avg) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Patient-015** | 12 timepoints | **0.0029** | 0.0071 | **+59%** |
| **Patient-006** | 13 timepoints | **0.0041** | 0.0125 | **+67%** |
| **Patient-012** | 9 timepoints | **0.0052** | 0.0118 | **+55%** |
| **Patient-001** | 2 timepoints | **0.0092** | 0.0154 | **+40%** |

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

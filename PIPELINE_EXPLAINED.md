# Glioblastoma Pipeline Explained

This document explains what the repository is doing now, how the current pipeline works, and what math the model is actually optimizing.

The main runnable script is:

- [`scripts/run_physics_pipeline.py`](/Users/tushar/Documents/Repositories/Glioblastoma/scripts/run_physics_pipeline.py)

The patient data used by the current experiments is:

- [`patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_007)
- [`patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/patient_067)

## 1. What Problem This Code Is Solving

The repository is trying to model longitudinal glioblastoma evolution from MRI scans collected at multiple weeks for the same patient.

At a high level, the task is:

1. Load a scan at time `t0`.
2. Estimate latent biological parameters from that scan.
3. Use a simplified growth equation to evolve the tumor forward in time.
4. Compare the predicted future tumor state to the real scan at time `t1`.

The current code does this with a **3D physics-informed model** rather than a pure image-to-image network.

The key idea is:

- The MRI gives anatomical evidence.
- A small neural network converts that anatomy into spatially varying biological parameters.
- A reaction-diffusion equation then simulates growth.

That is more structured than asking a large neural net to hallucinate the next scan directly.

## 2. What Data The Model Uses

Each patient folder contains 4 MRI modalities at several timepoints:

- `FLAIR`
- `T1`
- `T2`
- `CT1`

For example:

- `patient_007/FLAIR_wk055.nii`
- `patient_007/T1_wk055.nii`
- `patient_007/T2_wk055.nii`
- `patient_007/CT1_wk055.nii`

The current script only uses weeks where **all four modalities exist**.

For the present repository state:

- `patient_007` has weeks `55, 64, 75, 89, 105`
- `patient_067` has weeks `92, 109, 124, 136, 152`

That gives 4 adjacent time intervals per patient:

- `patient_007`: `55->64`, `64->75`, `75->89`, `89->105`
- `patient_067`: `92->109`, `109->124`, `124->136`, `136->152`

So in total:

- 8 adjacent training/evaluation pairs across both patients
- 4 adjacent pairs per patient when training each patient separately

That is a very small dataset. This matters for how much trust you should place in the results.

## 3. Preprocessing

The preprocessing is intentionally simple:

1. Load each NIfTI volume.
2. Convert it to `float32`.
3. Normalize each modality independently to `[0, 1]` using:

```text
x_norm = (x - min(x)) / (max(x) - min(x) + 1e-8)
```

4. Stack modalities into a 4-channel tensor.
5. Resize the full 3D tensor to a common grid, typically `64 x 64 x 64`.

So the model input looks like:

```text
x_t0 ∈ R^(4 x D x H x W)
```

where the 4 channels are the MRI modalities.

## 4. What The Model Predicts

The current model does **not** predict all modalities at the next timepoint.

Instead, it uses:

- the full 4-channel MRI at `t0` as input
- only the `FLAIR` channel at `t1` as the supervision target

In code, the current tumor state proxy is:

```text
c(t0) = FLAIR(t0)
```

This is an approximation. It assumes the normalized FLAIR image is a surrogate for tumor concentration. That is biologically crude, but it is the current working assumption.

## 5. Core Modeling Idea

The model has two parts:

1. A neural network that estimates parameter maps from the MRI.
2. A PDE-inspired solver that evolves the tumor concentration through time.

### 5.1 Parameter Estimator

The estimator takes the 4 MRI channels and predicts two 3D maps:

- `D(x)`: diffusion or invasion tendency
- `rho(x)`: proliferation or local growth tendency

Mathematically:

```text
(D, rho) = f_theta(x_t0)
```

where:

- `f_theta` is a small 3D convolutional network
- `theta` are the learned weights

The current script supports two sizes:

- `standard`: a 2-layer 3D convolutional encoder
- `tiny`: a much smaller 1-layer 3D encoder

The outputs are constrained to be positive and bounded:

```text
D(x)   = 0.5 * sigmoid(raw_D(x))
rho(x) = 0.5 * sigmoid(raw_rho(x))
```

So both maps live in:

```text
[0, 0.5]
```

That prevents the solver from learning obviously nonsensical negative rates.

## 6. The Growth Equation

The growth model is a simplified reaction-diffusion equation.

The intended continuous-time equation is:

```math
\frac{\partial c}{\partial t} = D(x)\nabla^2 c + \rho(x)c(1-c)
```

where:

- `c(x,t)` is tumor concentration
- `D(x)` controls spatial spread
- `rho(x)` controls local proliferation
- `c(1-c)` is logistic growth

Interpretation:

- The **diffusion term** spreads tumor outward.
- The **reaction term** lets tumor grow locally but saturates as `c -> 1`.

### 6.1 Why `c(1-c)`?

If growth were just proportional to `c`, it could explode without bound.

Using logistic growth:

```math
\rho c (1-c)
```

means:

- when `c` is small, growth is approximately linear
- when `c` approaches 1, growth slows down

This is a standard bounded-growth heuristic.

## 7. How The PDE Is Discretized

The solver does not integrate the PDE analytically. It uses an explicit finite-difference style update.

### 7.1 Laplacian Approximation

The code uses a 3D 6-neighbor Laplacian stencil:

```text
center = -6
each axis-aligned neighbor = +1
```

So the discrete Laplacian is:

```math
\nabla^2 c(i,j,k) \approx
c_{i+1,j,k} + c_{i-1,j,k} +
c_{i,j+1,k} + c_{i,j-1,k} +
c_{i,j,k+1} + c_{i,j,k-1} -
6c_{i,j,k}
```

That stencil is implemented as a 3D convolution.

### 7.2 Time Stepping

If the total physical time gap between scans is `dt`, the solver splits it into `N` smaller steps:

```math
\Delta t = \frac{dt}{N}
```

Then for each step:

```math
c^{k+1} =
\operatorname{clip}
\left(
c^k + \Delta t \left[D \odot \nabla^2 c^k + \rho \odot c^k \odot (1-c^k)\right],
0, 1
\right)
```

where:

- `⊙` means elementwise multiplication
- `clip(., 0, 1)` enforces valid concentration bounds

This is explicit Euler integration.

It is simple, differentiable, and easy to backpropagate through, but it is not a sophisticated numerical solver.

## 8. Therapy Simulation

The therapy visualization uses the same growth equation but adds a treatment term.

The modified update is:

```math
\frac{\partial c}{\partial t}
= D(x)\nabla^2 c + \rho(x)c(1-c) - T(x)c
```

where:

- `T(x)` is a hand-made radiation field
- it is implemented as a spherical mask centered at the current tumor maximum

In the code, this becomes:

```math
\text{kill} = T(x) \odot c \cdot \alpha
```

with an effectiveness constant `alpha`, currently set to `2.0`.

This is a toy treatment simulator, not a realistic radiotherapy model.

## 9. Training Objective

The loss has three parts.

### 9.1 Fit Loss

The main term is mean squared error between the predicted next FLAIR and the actual next FLAIR:

```math
L_{\text{fit}} = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2
```

This is the main supervision term.

### 9.2 Smoothness Loss

The model also penalizes abrupt changes in the learned parameter maps:

```math
L_{\text{smooth}} = TV(D) + TV(\rho)
```

where `TV` here is a simple average absolute finite-difference penalty along each spatial axis.

In practice, this encourages neighboring voxels to have similar biology unless the model has a reason not to.

### 9.3 Variance Penalty

There is also a term that discourages the diffusion map from collapsing into an almost constant field:

```math
L_{\text{var}} = \max(0, \tau - \operatorname{std}(D))
```

with threshold `tau = 0.05`.

If the diffusion map has too little variation, the penalty activates.

### 9.4 Total Loss

The current total loss is:

```math
L = L_{\text{fit}} + 0.001L_{\text{smooth}} + 10.0L_{\text{var}}
```

This weighting is heuristic. It was chosen to:

- fit the next scan
- keep parameter maps spatially smooth
- avoid trivially flat `D` maps

## 10. Standard Model vs Tiny Model

The repository now supports two learned parameter estimators.

### 10.1 Standard Model

This is the default.

It uses:

- `Conv3d(4 -> 16)`
- `GroupNorm`
- `SiLU`
- `Conv3d(16 -> 32)`
- `GroupNorm`
- `SiLU`
- `Conv3d(32 -> 2)` to emit `D` and `rho`

This has more capacity and tends to fit training pairs more aggressively.

### 10.2 Tiny Model

The tiny version uses:

- `Conv3d(4 -> 8)`
- `GroupNorm`
- `SiLU`
- `Conv3d(8 -> 2)`

This is much smaller and is intended to reduce overfitting on tiny data.

Because this dataset is so small, the tiny model is often the more defensible choice, even if it does not always win numerically.

## 11. Holdout Strategy

The script supports:

```text
--holdout-last-pair
```

For each patient, it removes the latest adjacent pair from training.

For example:

- `patient_007`: hold out `89 -> 105`
- `patient_067`: hold out `136 -> 152`

When running patients separately, each patient therefore has:

- 3 training pairs
- 1 holdout pair

This is still extremely small, but it is more honest than training on every pair and only reporting in-sample fit.

## 12. Baseline

The script also computes a simple persistence baseline:

```text
predict next FLAIR = current FLAIR
```

Mathematically:

```math
\hat{y}_{\text{baseline}} = c(t0)
```

This baseline matters because on short longitudinal intervals, MRI volumes often change less than people expect. A learned model needs to beat this simple rule to justify its complexity.

## 13. Metrics

The current summaries report two metrics.

### 13.1 MSE

```math
\text{MSE} = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2
```

This measures voxelwise intensity mismatch.

### 13.2 Relative Volume Difference

The model thresholds predicted and target FLAIR volumes at `0.5`, counts the number of above-threshold voxels, and compares volumes:

```math
V_{\text{pred}} = \sum_i \mathbf{1}(\hat{y}_i > 0.5)
```

```math
V_{\text{true}} = \sum_i \mathbf{1}(y_i > 0.5)
```

```math
\text{RelVolDiff} = \frac{|V_{\text{pred}} - V_{\text{true}}|}{\max(V_{\text{true}}, 1)}
```

This is a crude tumor-extent proxy, not a true segmentation metric.

## 14. What The Current Results Mean

The current experiments show a few consistent patterns:

1. The model can fit adjacent scan pairs at low MSE.
2. Holdout behavior is inconsistent across patients.
3. A smaller model sometimes helps, but not reliably.
4. The learned model does not dominate the persistence baseline in a strong, stable way.

This means:

- the pipeline is working as code
- the model is learning something
- but the evidence is too weak to claim robust prediction

That is because the limiting factor is not just architecture quality. It is data size.

## 15. Why This Data Is Not Enough For Strong Claims

With the current repository:

- 2 patients
- 5 timepoints each
- 4 adjacent intervals per patient

That is enough for:

- proof-of-concept experiments
- checking whether the pipeline runs
- seeing whether a structured model behaves sensibly

It is not enough for:

- robust generalization claims
- stable patient-to-patient conclusions
- clinically meaningful forecasting

When training patients separately, the issue becomes even sharper:

- each model only sees 3 training pairs before evaluation

That is far below what would normally be considered sufficient for a trustworthy deep-learning model.

## 16. What Files The Script Produces

For each run, the script writes:

- a checkpoint
- a loss curve
- forecast visualization
- therapy visualization
- a JSON summary with metrics

Example run directories:

- [`runs/physics_run_separate_holdout_64_patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/physics_run_separate_holdout_64_patient_007)
- [`runs/physics_run_separate_holdout_64_patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/physics_run_separate_holdout_64_patient_067)
- [`runs/physics_run_tiny_baseline_holdout_64_patient_007`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/physics_run_tiny_baseline_holdout_64_patient_007)
- [`runs/physics_run_tiny_baseline_holdout_64_patient_067`](/Users/tushar/Documents/Repositories/Glioblastoma/runs/physics_run_tiny_baseline_holdout_64_patient_067)

## 17. Important Limitations

The current pipeline has several strong simplifying assumptions:

1. `FLAIR` is treated as tumor concentration.
2. The PDE uses `D(x) * Laplacian(c)` instead of a more complete spatial diffusion operator.
3. No segmentation labels are used.
4. No tissue-specific anatomy constraints are used.
5. The therapy simulation is a synthetic spherical radiation field.
6. The numerical solver is explicit Euler, which is simple but not especially sophisticated.
7. Evaluation is based on very few held-out examples.

So this should be interpreted as a structured exploratory model, not a validated biomedical forecasting system.

## 18. Short Summary

The current repository is doing the following:

1. Build 3D MRI pairs from longitudinal patient scans.
2. Estimate spatial growth parameters from MRI with a small CNN.
3. Evolve tumor concentration forward using a reaction-diffusion equation.
4. Train by matching the next FLAIR volume.
5. Evaluate on held-out adjacent pairs and compare against a persistence baseline.

The math is simple enough to inspect, which is the main advantage of this approach.
The data is far too small for strong predictive claims, which is the main limitation.

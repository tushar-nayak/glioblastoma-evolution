from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchdiffeq import odeint


MODALITIES = ("FLAIR", "T1", "T2", "CT1")
WEEK_PATTERN = re.compile(r"wk(\d+)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the history-conditioned attention U-Net Neural ODE approach on the local patient folders."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to the directory containing patient data.")
    parser.add_argument("--lumiere", action="store_true", help="Whether the dataset is in the LUMIERE structure.")
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--history-mode",
        choices=("prefix", "sliding"),
        default="prefix",
        help="Use full prefix history for each target week or a fixed sliding context window.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=3,
        help="Number of context weeks per sample when history-mode=sliding.",
    )
    parser.add_argument(
        "--slice-offsets",
        nargs="*",
        type=int,
        default=[-1, 0, 1],
        help="Relative slice offsets around the center slice for each week.",
    )
    parser.add_argument(
        "--model-size",
        choices=("standard", "tiny"),
        default="standard",
        help="Backbone size for the attention U-Net Neural ODE.",
    )
    parser.add_argument(
        "--holdout-last-pair",
        action="store_true",
        help="Hold out the latest target week for each patient from training.",
    )
    parser.add_argument("--separate-patient-runs", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_week(filename: str) -> int:
    match = WEEK_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"Could not parse week from {filename}")
    return int(match.group(1))


@dataclass
class ForecastSample:
    patient_id: str
    history_weeks: list[int]
    target_week: int
    dt_years: float


class HistoryForecastDataset(Dataset):
    def __init__(
        self,
        patient_dirs: list[Path],
        history_mode: str = "prefix",
        context_size: int = 3,
        slice_offsets: list[int] | tuple[int, ...] = (-1, 0, 1),
        is_lumiere: bool = False,
    ) -> None:
        self.patient_dirs = [path.resolve() for path in patient_dirs]
        self.patient_dirs_by_name = {path.name: path.resolve() for path in self.patient_dirs}
        self.history_mode = history_mode
        self.context_size = context_size
        self.slice_offsets = tuple(slice_offsets)
        self.is_lumiere = is_lumiere
        self.patient_weeks: dict[str, list[int]] = {}
        self.patient_files: dict[str, dict[int, dict[str, Path]]] = {}
        self.slice_cache: dict[tuple[str, int], torch.Tensor] = {}
        self.registration_cache: dict[tuple[str, int, str], np.ndarray] = {}
        self.samples: list[ForecastSample] = []

        for patient_dir in self.patient_dirs:
            weeks = self._discover_weeks(patient_dir)
            self.patient_weeks[patient_dir.name] = weeks
            if len(weeks) < 2:
                print(f"Skipping patient {patient_dir.name}: only {len(weeks)} weeks found.")
                continue

            if self.history_mode == "prefix":
                for target_idx in range(1, len(weeks)):
                    history_weeks = weeks[:target_idx]
                    target_week = weeks[target_idx]
                    self.samples.append(
                        ForecastSample(
                            patient_id=patient_dir.name,
                            history_weeks=list(history_weeks),
                            target_week=target_week,
                            dt_years=(target_week - history_weeks[-1]) / 52.0,
                        )
                    )
            else:
                if len(weeks) < self.context_size + 1:
                    print(f"Skipping patient {patient_dir.name}: not enough weeks for sliding context.")
                    continue
                for start_idx in range(0, len(weeks) - self.context_size):
                    history_weeks = weeks[start_idx : start_idx + self.context_size]
                    for target_week in weeks[start_idx + self.context_size :]:
                        self.samples.append(
                            ForecastSample(
                                patient_id=patient_dir.name,
                                history_weeks=list(history_weeks),
                                target_week=target_week,
                                dt_years=(target_week - history_weeks[-1]) / 52.0,
                            )
                        )

        if not self.samples:
            raise RuntimeError("No valid history-conditioned training samples were found.")
        self.pairs = self.samples

    def _discover_weeks(self, patient_dir: Path) -> list[int]:
        week_to_paths: dict[int, dict[str, Path]] = {}
        if self.is_lumiere:
            # LUMIERE structure: Patient-XXX/week-YYY/DeepBraTumIA-segmentation/atlas/skull_strip/
            # and files are flair_skull_strip.nii.gz, etc.
            for week_dir in patient_dir.glob("week-*"):
                try:
                    week_num = int(week_dir.name.split("-")[1])
                except (IndexError, ValueError):
                    continue
                
                skull_strip_path = week_dir / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
                if not skull_strip_path.exists():
                    continue
                
                paths = {}
                mapping = {"FLAIR": "flair", "T1": "t1", "T2": "t2", "CT1": "ct1"}
                for mod_key, mod_file_prefix in mapping.items():
                    matches = list(skull_strip_path.glob(f"{mod_file_prefix}_skull_strip.nii*"))
                    if matches:
                        paths[mod_key] = matches[0]
                
                if all(mod in paths for mod in MODALITIES):
                    week_to_paths[week_num] = paths
        else:
            for mod in MODALITIES:
                for mod_file in sorted(patient_dir.glob(f"{mod}_wk*.nii")):
                    week = extract_week(mod_file.name)
                    week_to_paths.setdefault(week, {})[mod] = mod_file
        
        valid_week_map = {
            week: paths
            for week, paths in week_to_paths.items()
            if all(mod in paths for mod in MODALITIES)
        }
        weeks = sorted(valid_week_map)
        self.patient_files[patient_dir.name] = valid_week_map
        return weeks

    def _register_to_reference(self, moving_vol: np.ndarray, fixed_vol: np.ndarray) -> np.ndarray:
        # Robust Python equivalent of MATLAB imregister(moving, fixed, "affine") using SimpleITK
        import SimpleITK as sitk
        
        fixed_image = sitk.GetImageFromArray(fixed_vol)
        moving_image = sitk.GetImageFromArray(moving_vol)

        # Basic registration configuration
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric
        registration_method.SetMetricAsMeanSquares()
        
        # Optimizer
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, 
            minStep=1e-4, 
            numberOfIterations=100
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        try:
            final_transform = registration_method.Execute(fixed_image, moving_image)
            resampled = sitk.Resample(
                moving_image, 
                fixed_image, 
                final_transform, 
                sitk.sitkLinear, 
                0.0, 
                moving_image.GetPixelID()
            )
            return sitk.GetArrayFromImage(resampled)
        except Exception as e:
            print(f"Registration failed, returning original volume: {e}")
            return moving_vol

    def _load_week_slices(self, patient_id: str, week: int, reference_week: int | None = None) -> torch.Tensor:
        cache_key = (patient_id, week, reference_week)
        cached = self.slice_cache.get(cache_key)
        if cached is not None:
            return cached

        # In LUMIERE, we might want to register all weeks to the latest (or target) week's CT1
        # as seen in the user's MATLAB code.
        
        modality_volumes = {}
        for mod in MODALITIES:
            img_path = self.patient_files[patient_id][week][mod]
            volume = nib.load(img_path).get_fdata().astype(np.float32)
            # Basic normalization
            volume = (volume - np.min(volume)) / (max(np.max(volume) - np.min(volume), 1e-8))
            modality_volumes[mod] = volume

        # Registration step if a reference week is provided and it's different from the current week
        if reference_week is not None and reference_week != week:
            ref_path = self.patient_files[patient_id][reference_week]["CT1"]
            ref_vol = nib.load(ref_path).get_fdata().astype(np.float32)
            ref_vol = (ref_vol - np.min(ref_vol)) / (max(np.max(ref_vol) - np.min(ref_vol), 1e-8))
            
            for mod in MODALITIES:
                modality_volumes[mod] = self._register_to_reference(modality_volumes[mod], ref_vol)

        modality_slices = []
        for mod in MODALITIES:
            volume = modality_volumes[mod]
            h, w = volume.shape[:2]
            h = (h // 16) * 16
            w = (w // 16) * 16
            volume = volume[:h, :w, :]

            center_slice = volume.shape[2] // 2
            selected_slices = []
            for offset in self.slice_offsets:
                slice_idx = min(max(center_slice + offset, 0), volume.shape[2] - 1)
                selected_slices.append(torch.from_numpy(volume[:, :, slice_idx]))
            modality_slices.append(torch.stack(selected_slices))

        week_tensor = torch.stack(modality_slices)  # [4, S, H, W]
        # Cache the result to avoid re-registering in future epochs
        self.slice_cache[cache_key] = week_tensor
        return week_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        history_week_tensors = []
        
        # User registers to the latest scan (target_week)
        for week in sample.history_weeks:
            week_tensor = self._load_week_slices(sample.patient_id, week, reference_week=sample.target_week)
            history_week_tensors.append(week_tensor.reshape(-1, week_tensor.shape[2], week_tensor.shape[3]))

        history_tensor = torch.stack(history_week_tensors, dim=0)
        target_tensor = self._load_week_slices(sample.patient_id, sample.target_week)[:, len(self.slice_offsets) // 2]
        return {
            "history": history_tensor,
            "target": target_tensor,
            "time_delta": torch.tensor(sample.dt_years, dtype=torch.float32),
            "history_weeks": torch.tensor(sample.history_weeks, dtype=torch.long),
            "patient_id": sample.patient_id,
            "target_week": sample.target_week,
        }


NeuralODEDataset = HistoryForecastDataset


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class AttentionGate(nn.Module):
    def __init__(self, g_channels: int, s_channels: int, out_channels: int) -> None:
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(g_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.ws = nn.Sequential(nn.Conv2d(s_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(nn.Conv2d(out_channels, 1, 1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=True)
        attention = self.output(self.relu(self.wg(g) + self.ws(s)))
        return attention * s


class DecoderBlock(nn.Module):
    def __init__(self, g_channels: int, s_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention = AttentionGate(g_channels, s_channels, out_channels)
        self.conv = ConvBlock(g_channels + s_channels, out_channels)

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        g = self.up(g)
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=True)
        s = self.attention(g, s)
        return self.conv(torch.cat([g, s], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: list[int]) -> None:
        super().__init__()
        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        self.decoder1 = DecoderBlock(features[3] * 2, features[3], features[3])
        self.decoder2 = DecoderBlock(features[3], features[2], features[2])
        self.decoder3 = DecoderBlock(features[2], features[1], features[1])
        self.decoder4 = DecoderBlock(features[1], features[0], features[0])
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        b = self.bottleneck(p4)
        d1 = self.decoder1(b, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)
        return self.final_conv(d4)


class TimeAwareODEFunc(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, hidden_dim),
        )
        groups = 8 if hidden_dim >= 8 else 1
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_vector = self.time_embed(t.view(1, 1).to(dtype=y.dtype, device=y.device))
        t_expanded = t_vector.view(1, -1, 1, 1).expand(y.size(0), -1, y.size(2), y.size(3))
        return self.conv_block(torch.cat([y, t_expanded], dim=1))


class TemporalEncoder(nn.Module):
    def __init__(self, max_weeks: int, hidden_dim: int) -> None:
        super().__init__()
        self.week_embed = nn.Embedding(max_weeks + 1, hidden_dim)

    def forward(self, week_numbers: torch.Tensor) -> torch.Tensor:
        return self.week_embed(week_numbers.long())


class HistoryConditionedGliomaNeuralODEModel(nn.Module):
    def __init__(
        self,
        week_input_channels: int,
        output_channels: int,
        hidden_dim: int,
        features: list[int],
        max_weeks: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.week_input_channels = week_input_channels
        self.unet = AttentionUNet(week_input_channels, hidden_dim, features=features)
        self.temporal_encoder = TemporalEncoder(max_weeks=max_weeks, hidden_dim=hidden_dim)
        self.ode_func = TimeAwareODEFunc(hidden_dim)
        groups = 8 if hidden_dim >= 8 else 1
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, output_channels, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        t_delta: torch.Tensor,
        history_weeks: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, history_len, _, _, _ = history.shape
        week_features = history.reshape(
            batch_size * history_len,
            history.size(2),
            history.size(3),
            history.size(4),
        )
        encoded = self.unet(week_features)
        encoded = encoded.reshape(batch_size, history_len, self.hidden_dim, encoded.size(-2), encoded.size(-1))

        temporal_emb = self.temporal_encoder(history_weeks.clamp_min(0).to(history.device))
        temporal_emb = temporal_emb.view(batch_size, history_len, self.hidden_dim, 1, 1)

        mask = history_mask.to(history.device, dtype=encoded.dtype).view(batch_size, history_len, 1, 1, 1)
        encoded = (encoded + temporal_emb) * mask
        hidden = encoded.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        outputs = []
        for b in range(batch_size):
            t_span = torch.linspace(
                0.0,
                float(t_delta[b].item()),
                steps=4,
                dtype=torch.float32,
                device=history.device,
            )
            evolved = odeint(self.ode_func, hidden[b : b + 1], t_span, method="midpoint")
            outputs.append(torch.sigmoid(self.decoder(evolved[-1])))
        return torch.cat(outputs, dim=0)


GliomaNeuralODEModel = HistoryConditionedGliomaNeuralODEModel


def build_holdout_last_pair_split(dataset: NeuralODEDataset) -> tuple[list[int], list[int]]:
    latest_target_by_patient = {
        patient_id: max(weeks) for patient_id, weeks in dataset.patient_weeks.items()
    }
    holdout_indices = sorted(
        idx
        for idx, sample in enumerate(dataset.samples)
        if sample.target_week == latest_target_by_patient[sample.patient_id]
    )
    train_indices = [idx for idx in range(len(dataset.samples)) if idx not in holdout_indices]
    if not train_indices:
        raise RuntimeError("Holdout split consumed all history-conditioned training samples.")
    return train_indices, holdout_indices


def evaluate_model(
    model: GliomaNeuralODEModel,
    dataset: NeuralODEDataset,
    device: torch.device,
    indices: list[int] | None = None,
) -> list[dict[str, float | int | str]]:
    indices = indices if indices is not None else list(range(len(dataset.samples)))
    mse_crit = nn.MSELoss()
    mae_crit = nn.L1Loss()
    metrics = []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            sample_meta = dataset.samples[idx]
            sample = dataset[idx]
            history = sample["history"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            time_delta = sample["time_delta"].unsqueeze(0).to(device)
            history_weeks = sample["history_weeks"].unsqueeze(0).to(device)
            history_mask = torch.ones((1, history.shape[1]), dtype=torch.bool, device=device)
            prediction = model(history, time_delta, history_weeks, history_mask)
            mse = float(mse_crit(prediction, target).item())
            mae = float(mae_crit(prediction, target).item())
            per_modality_mse = {
                mod: float(mse_crit(prediction[:, i : i + 1], target[:, i : i + 1]).item())
                for i, mod in enumerate(MODALITIES)
            }
            per_modality_mae = {
                mod: float(mae_crit(prediction[:, i : i + 1], target[:, i : i + 1]).item())
                for i, mod in enumerate(MODALITIES)
            }

            pred_flair = prediction[0, 0].detach().cpu().numpy()
            true_flair = target[0, 0].detach().cpu().numpy()
            pred_volume = float((pred_flair > 0.5).sum())
            true_volume = float((true_flair > 0.5).sum())
            volume_diff = abs(pred_volume - true_volume) / max(true_volume, 1.0)

            metrics.append(
                {
                    "patient_id": sample_meta.patient_id,
                    "history_weeks": sample_meta.history_weeks,
                    "target_week": sample_meta.target_week,
                    "mse": mse,
                    "mae": mae,
                    "per_modality_mse": per_modality_mse,
                    "per_modality_mae": per_modality_mae,
                    "relative_flair_volume_diff": volume_diff,
                }
            )
    return metrics


def evaluate_persistence_baseline(
    dataset: NeuralODEDataset,
    indices: list[int] | None = None,
) -> list[dict[str, float | int | str]]:
    indices = indices if indices is not None else list(range(len(dataset.samples)))
    mse_crit = nn.MSELoss()
    mae_crit = nn.L1Loss()
    metrics = []
    for idx in indices:
        sample_meta = dataset.samples[idx]
        sample = dataset[idx]
        latest_history = sample["history"][-1].reshape(
            len(MODALITIES),
            len(dataset.slice_offsets),
            sample["history"].shape[2],
            sample["history"].shape[3],
        )
        baseline_prediction = latest_history[:, len(dataset.slice_offsets) // 2].unsqueeze(0)
        target = sample["target"].unsqueeze(0)
        mse = float(mse_crit(baseline_prediction, target).item())
        mae = float(mae_crit(baseline_prediction, target).item())
        per_modality_mse = {
            mod: float(mse_crit(baseline_prediction[:, i : i + 1], target[:, i : i + 1]).item())
            for i, mod in enumerate(MODALITIES)
        }
        per_modality_mae = {
            mod: float(mae_crit(baseline_prediction[:, i : i + 1], target[:, i : i + 1]).item())
            for i, mod in enumerate(MODALITIES)
        }
        pred_flair = baseline_prediction[0, 0].detach().cpu().numpy()
        true_flair = target[0, 0].detach().cpu().numpy()
        pred_volume = float((pred_flair > 0.5).sum())
        true_volume = float((true_flair > 0.5).sum())
        volume_diff = abs(pred_volume - true_volume) / max(true_volume, 1.0)
        metrics.append(
            {
                "patient_id": sample_meta.patient_id,
                "history_weeks": sample_meta.history_weeks,
                "target_week": sample_meta.target_week,
                "mse": mse,
                "mae": mae,
                "per_modality_mse": per_modality_mse,
                "per_modality_mae": per_modality_mae,
                "relative_flair_volume_diff": volume_diff,
            }
        )
    return metrics


def summarize_metric_rows(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    if not rows:
        return {
            "count": 0,
            "avg_mse": None,
            "avg_mae": None,
            "avg_relative_flair_volume_diff": None,
            "avg_per_modality_mse": {},
            "avg_per_modality_mae": {},
            "by_patient": {},
        }
    by_patient: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_patient.setdefault(str(row["patient_id"]), []).append(row)
    patient_summary = {}
    for patient_id, patient_rows in by_patient.items():
        avg_per_modality_mse = {
            mod: float(
                sum(float(r["per_modality_mse"][mod]) for r in patient_rows) / len(patient_rows)
            )
            for mod in MODALITIES
        }
        avg_per_modality_mae = {
            mod: float(
                sum(float(r["per_modality_mae"][mod]) for r in patient_rows) / len(patient_rows)
            )
            for mod in MODALITIES
        }
        patient_summary[patient_id] = {
            "count": len(patient_rows),
            "avg_mse": float(sum(float(r["mse"]) for r in patient_rows) / len(patient_rows)),
            "avg_mae": float(sum(float(r["mae"]) for r in patient_rows) / len(patient_rows)),
            "avg_relative_flair_volume_diff": float(
                sum(float(r["relative_flair_volume_diff"]) for r in patient_rows) / len(patient_rows)
            ),
            "avg_per_modality_mse": avg_per_modality_mse,
            "avg_per_modality_mae": avg_per_modality_mae,
        }
    avg_per_modality_mse = {
        mod: float(sum(float(r["per_modality_mse"][mod]) for r in rows) / len(rows))
        for mod in MODALITIES
    }
    avg_per_modality_mae = {
        mod: float(sum(float(r["per_modality_mae"][mod]) for r in rows) / len(rows))
        for mod in MODALITIES
    }
    return {
        "count": len(rows),
        "avg_mse": float(sum(float(r["mse"]) for r in rows) / len(rows)),
        "avg_mae": float(sum(float(r["mae"]) for r in rows) / len(rows)),
        "avg_relative_flair_volume_diff": float(
            sum(float(r["relative_flair_volume_diff"]) for r in rows) / len(rows)
        ),
        "avg_per_modality_mse": avg_per_modality_mse,
        "avg_per_modality_mae": avg_per_modality_mae,
        "by_patient": patient_summary,
    }


def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
    ax.set_title("Neural ODE Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prediction(
    sample: dict[str, object],
    prediction: torch.Tensor,
    output_path: Path,
    slice_count: int,
) -> None:
    history = sample["history"]
    target = sample["target"]
    latest_history = history[-1].reshape(len(MODALITIES), slice_count, history.shape[2], history.shape[3])
    fig, ax = plt.subplots(3, len(MODALITIES), figsize=(4 * len(MODALITIES), 10))
    for i, mod in enumerate(MODALITIES):
        latest_history_img = latest_history[i, slice_count // 2].detach().cpu().numpy()
        pred_img = prediction[0, i].detach().cpu().numpy()
        target_img = target[i].detach().cpu().numpy()

        ax[0, i].imshow(latest_history_img, cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title(f"Latest History {mod}")
        ax[1, i].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
        ax[1, i].set_title(f"Prediction {mod}")
        ax[2, i].imshow(target_img, cmap="gray", vmin=0, vmax=1)
        ax[2, i].set_title(f"Target {mod}")
        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")
    history_weeks = sample["history_weeks"].tolist()
    target_week = int(sample["target_week"])
    fig.suptitle(f"History weeks {history_weeks} -> target week {target_week}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def collate_history_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    max_history = max(item["history"].shape[0] for item in batch)
    history_channels = batch[0]["history"].shape[1]
    height = batch[0]["history"].shape[2]
    width = batch[0]["history"].shape[3]

    histories = []
    history_weeks = []
    history_masks = []
    targets = []
    time_deltas = []
    patient_ids = []
    target_weeks = []

    for item in batch:
        history = item["history"]
        history_len = history.shape[0]
        pad_len = max_history - history_len
        if pad_len > 0:
            pad = torch.zeros(
                pad_len,
                history_channels,
                height,
                width,
                dtype=history.dtype,
            )
            history = torch.cat([history, pad], dim=0)

        weeks = item["history_weeks"]
        if pad_len > 0:
            weeks = torch.cat([weeks, torch.full((pad_len,), -1, dtype=weeks.dtype)], dim=0)

        mask = torch.zeros(max_history, dtype=torch.bool)
        mask[:history_len] = True

        histories.append(history)
        history_weeks.append(weeks)
        history_masks.append(mask)
        targets.append(item["target"])
        time_deltas.append(item["time_delta"])
        patient_ids.append(item["patient_id"])
        target_weeks.append(item["target_week"])

    return {
        "history": torch.stack(histories, dim=0),
        "history_weeks": torch.stack(history_weeks, dim=0),
        "history_mask": torch.stack(history_masks, dim=0),
        "target": torch.stack(targets, dim=0),
        "time_delta": torch.stack(time_deltas, dim=0),
        "patient_id": patient_ids,
        "target_week": target_weeks,
    }


def train_model(
    model: GliomaNeuralODEModel,
    dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[float], list[dict[str, float]]]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_history_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()

    epoch_losses = []
    epoch_details = []

    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_mse = 0.0
        running_l1 = 0.0
        for batch in dataloader:
            history = batch["history"].to(device)
            target = batch["target"].to(device)
            time_delta = batch["time_delta"].to(device)
            history_weeks = batch["history_weeks"].to(device)
            history_mask = batch["history_mask"].to(device)

            optimizer.zero_grad()
            prediction = model(history, time_delta, history_weeks, history_mask)
            mse_loss = mse_crit(prediction, target)
            l1_loss = l1_crit(prediction, target)
            total_loss = mse_loss + 0.1 * l1_loss
            total_loss.backward()
            optimizer.step()

            running_total += float(total_loss.item())
            running_mse += float(mse_loss.item())
            running_l1 += float(l1_loss.item())

        num_batches = len(dataloader)
        epoch_losses.append(running_total / num_batches)
        epoch_details.append(
            {
                "epoch": epoch + 1,
                "total_loss": running_total / num_batches,
                "mse_loss": running_mse / num_batches,
                "l1_loss": running_l1 / num_batches,
            }
        )
        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"loss={running_total / num_batches:.6f} | "
            f"mse={running_mse / num_batches:.6f}"
        )

    return epoch_losses, epoch_details


def build_model(
    week_input_channels: int,
    output_channels: int,
    max_weeks: int,
    model_size: str,
) -> GliomaNeuralODEModel:
    if model_size == "tiny":
        hidden_dim = 32
        features = [16, 32, 64, 128]
    else:
        hidden_dim = 64
        features = [32, 64, 128, 256]
    return GliomaNeuralODEModel(
        week_input_channels=week_input_channels,
        output_channels=output_channels,
        hidden_dim=hidden_dim,
        features=features,
        max_weeks=max_weeks,
    )


def run_experiment(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    patient_names: list[str],
    run_name: str,
    data_dir: Path | None = None,
    is_lumiere: bool = False,
) -> Path:
    search_root = data_dir or repo_root
    patient_dirs = [search_root / name for name in patient_names]
    missing = [path for path in patient_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing patient directories: {missing}")

    device = get_device()
    run_dir = repo_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Patients: {patient_names}")

    dataset = NeuralODEDataset(
        patient_dirs=patient_dirs,
        history_mode=args.history_mode,
        context_size=args.context_size,
        slice_offsets=args.slice_offsets,
        is_lumiere=is_lumiere,
    )
    print(f"History-conditioned samples: {len(dataset)}")
    print(f"Patient weeks: {dataset.patient_weeks}")

    train_indices = list(range(len(dataset.samples)))
    holdout_indices: list[int] = []
    if args.holdout_last_pair:
        train_indices, holdout_indices = build_holdout_last_pair_split(dataset)
        print(f"Train samples after holdout: {len(train_indices)}")
        print(f"Holdout samples: {len(holdout_indices)}")

    train_dataset: Dataset = Subset(dataset, train_indices) if holdout_indices else dataset
    sample = dataset[0]
    model = build_model(
        week_input_channels=int(sample["history"].shape[1]),
        output_channels=int(sample["target"].shape[0]),
        max_weeks=max(max(weeks) for weeks in dataset.patient_weeks.values()),
        model_size=args.model_size,
    ).to(device)

    losses, epoch_details = train_model(
        model=model,
        dataset=train_dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = run_dir / f"attention_unet_neural_ode_{args.model_size}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    plot_loss_curve(losses, run_dir / "loss_curve.png")

    train_metrics = evaluate_model(model, dataset, device, indices=train_indices)
    holdout_metrics = evaluate_model(model, dataset, device, indices=holdout_indices) if holdout_indices else []
    all_metrics = evaluate_model(model, dataset, device)
    baseline_train_metrics = evaluate_persistence_baseline(dataset, indices=train_indices)
    baseline_holdout_metrics = (
        evaluate_persistence_baseline(dataset, indices=holdout_indices) if holdout_indices else []
    )
    baseline_all_metrics = evaluate_persistence_baseline(dataset)

    print(f"Model holdout summary: {summarize_metric_rows(holdout_metrics)}")
    print(f"Baseline holdout summary: {summarize_metric_rows(baseline_holdout_metrics)}")

    summary_predictions = []
    with torch.no_grad():
        prediction_indices = holdout_indices or [len(dataset.samples) - 1]
        for idx in prediction_indices:
            sample = dataset[idx]
            patient_id = str(sample["patient_id"])
            patient_run_dir = run_dir / patient_id
            patient_run_dir.mkdir(parents=True, exist_ok=True)
            history = sample["history"].unsqueeze(0).to(device)
            time_delta = sample["time_delta"].unsqueeze(0).to(device)
            history_weeks = sample["history_weeks"].unsqueeze(0).to(device)
            history_mask = torch.ones((1, history.shape[1]), dtype=torch.bool, device=device)
            prediction = model(history, time_delta, history_weeks, history_mask)
            output_path = patient_run_dir / f"prediction_to_week_{int(sample['target_week'])}.png"
            plot_prediction(
                sample=sample,
                prediction=prediction,
                output_path=output_path,
                slice_count=len(args.slice_offsets),
            )
            summary_predictions.append(
                {
                    "patient_id": patient_id,
                    "history_weeks": sample["history_weeks"].tolist(),
                    "target_week": int(sample["target_week"]),
                    "visualization": str(output_path),
                }
            )

    metadata = {
        "run_name": run_name,
        "repo_root": str(repo_root),
        "device": str(device),
        "patients": patient_names,
        "patient_weeks": dataset.patient_weeks,
        "history_mode": args.history_mode,
        "context_size": args.context_size,
        "slice_offsets": args.slice_offsets,
        "num_samples": len(dataset),
        "train_sample_count": len(train_indices),
        "holdout_sample_count": len(holdout_indices),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_size": args.model_size,
        "holdout_last_pair": args.holdout_last_pair,
        "checkpoint": str(checkpoint_path),
        "epoch_details": epoch_details,
        "train_indices": train_indices,
        "holdout_indices": holdout_indices,
        "train_history_metrics": train_metrics,
        "holdout_history_metrics": holdout_metrics,
        "all_history_metrics": all_metrics,
        "train_metric_summary": summarize_metric_rows(train_metrics),
        "holdout_metric_summary": summarize_metric_rows(holdout_metrics),
        "all_metric_summary": summarize_metric_rows(all_metrics),
        "baseline_name": "latest_history_center_slice",
        "baseline_train_history_metrics": baseline_train_metrics,
        "baseline_holdout_history_metrics": baseline_holdout_metrics,
        "baseline_all_history_metrics": baseline_all_metrics,
        "baseline_train_metric_summary": summarize_metric_rows(baseline_train_metrics),
        "baseline_holdout_metric_summary": summarize_metric_rows(baseline_holdout_metrics),
        "baseline_all_metric_summary": summarize_metric_rows(baseline_all_metrics),
        "prediction_visualizations": summary_predictions,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved summary: {summary_path}")
    return run_dir


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    repo_root = args.repo_root.resolve()
    
    search_root = args.data_dir.resolve() if args.data_dir else repo_root
    if args.patients:
        patient_names = args.patients
    elif args.lumiere:
        patient_names = sorted(path.name for path in search_root.glob("Patient-*") if path.is_dir())
    else:
        patient_names = sorted(path.name for path in search_root.glob("patient_*") if path.is_dir())

    if args.separate_patient_runs:
        base_run_name = args.run_name or datetime.now().strftime("neural_ode_run_%Y%m%d_%H%M%S")
        run_dirs = []
        for patient_name in patient_names:
            print(f"\n=== Separate Neural ODE run for {patient_name} ===")
            run_dirs.append(
                run_experiment(
                    args=args,
                    repo_root=repo_root,
                    patient_names=[patient_name],
                    run_name=f"{base_run_name}_{patient_name}",
                    data_dir=args.data_dir,
                    is_lumiere=args.lumiere,
                )
            )
        print("\nCompleted Neural ODE runs:")
        for run_dir in run_dirs:
            print(run_dir)
        return

    run_name = args.run_name or datetime.now().strftime("neural_ode_run_%Y%m%d_%H%M%S")
    run_experiment(
        args=args,
        repo_root=repo_root,
        patient_names=patient_names,
        run_name=run_name,
        data_dir=args.data_dir,
        is_lumiere=args.lumiere,
    )


if __name__ == "__main__":
    main()

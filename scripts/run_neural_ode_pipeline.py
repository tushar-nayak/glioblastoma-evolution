from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
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
    parser.add_argument(
        "--registered-data-dir",
        type=Path,
        default=None,
        help="Optional path to pre-registered full NIfTI volumes keyed by patient, target week, and moving week.",
    )
    parser.add_argument("--lumiere", action="store_true", help="Whether the dataset is in the LUMIERE structure.")
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for cohort runs or patient fine-tuning.")
    parser.add_argument(
        "--cohort-pretrain-epochs",
        type=int,
        default=12,
        help="Cohort pretraining epochs before separate patient fine-tuning.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to use for training and evaluation. 'auto' prefers MPS, then CUDA, then CPU.",
    )
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
        default=[-2, -1, 0, 1, 2],
        help="Relative slice offsets around the center slice for each history week.",
    )
    parser.add_argument(
        "--target-slice-offsets",
        nargs="*",
        type=int,
        default=None,
        help="Relative slice offsets predicted at the target week. Defaults to --slice-offsets.",
    )
    parser.add_argument(
        "--model-size",
        choices=("standard", "tiny"),
        default="standard",
        help="Backbone size for the attention U-Net Neural ODE.",
    )
    parser.add_argument(
        "--holdout-last-pair",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda is not available.")
        return torch.device("cuda")
    if requested_device == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS was requested with --device mps, but torch MPS is not available.")
        return torch.device("mps")
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


def unique_sorted_offsets(*offset_groups: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    values: set[int] = set()
    for group in offset_groups:
        values.update(int(value) for value in group)
    return tuple(sorted(values))


def reshape_stacked_modalities(tensor: torch.Tensor, slice_count: int) -> torch.Tensor:
    if tensor.dim() == 3:
        channels, height, width = tensor.shape
        return tensor.reshape(len(MODALITIES), slice_count, height, width)
    if tensor.dim() == 4:
        batch, channels, height, width = tensor.shape
        return tensor.reshape(batch, len(MODALITIES), slice_count, height, width)
    raise ValueError(f"Unsupported tensor shape for modality reshape: {tuple(tensor.shape)}")


def flair_focus_mask(target: torch.Tensor, target_slice_count: int) -> torch.Tensor:
    target_modal = reshape_stacked_modalities(target, target_slice_count)
    flair_volume = target_modal[:, 0]
    flair_threshold = flair_volume.amax(dim=(1, 2, 3), keepdim=True) * 0.45
    focus = (flair_volume >= flair_threshold).float().amax(dim=1, keepdim=True)
    return focus


def gradient_magnitude(batch: torch.Tensor) -> torch.Tensor:
    kernel_x = batch.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = batch.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    batch_size, channels, height, width = batch.shape
    flat = batch.reshape(batch_size * channels, 1, height, width)
    grad_x = F.conv2d(flat, kernel_x, padding=1)
    grad_y = F.conv2d(flat, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    return magnitude.reshape(batch_size, channels, height, width)


def compute_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_slice_count: int,
) -> dict[str, torch.Tensor]:
    mse_loss = F.mse_loss(prediction, target)
    l1_loss = F.l1_loss(prediction, target)
    edge_loss = F.l1_loss(gradient_magnitude(prediction), gradient_magnitude(target))

    focus = flair_focus_mask(target, target_slice_count)
    focus_weights = 1.0 + 2.5 * focus
    focus_weights = focus_weights.repeat(1, prediction.shape[1], 1, 1)
    focus_l1 = (focus_weights * (prediction - target).abs()).mean()

    total_loss = mse_loss + 0.15 * l1_loss + 0.10 * edge_loss + 0.20 * focus_l1
    return {
        "total_loss": total_loss,
        "mse_loss": mse_loss,
        "l1_loss": l1_loss,
        "edge_loss": edge_loss,
        "focus_l1_loss": focus_l1,
    }


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
        slice_offsets: list[int] | tuple[int, ...] = (-2, -1, 0, 1, 2),
        target_slice_offsets: list[int] | tuple[int, ...] | None = None,
        is_lumiere: bool = False,
        registered_data_dir: Path | None = None,
    ) -> None:
        self.patient_dirs = [path.resolve() for path in patient_dirs]
        self.patient_dirs_by_name = {path.name: path.resolve() for path in self.patient_dirs}
        self.history_mode = history_mode
        self.context_size = context_size
        self.slice_offsets = tuple(slice_offsets)
        self.target_slice_offsets = tuple(target_slice_offsets or slice_offsets)
        self.all_slice_offsets = unique_sorted_offsets(self.slice_offsets, self.target_slice_offsets)
        self.slice_offset_to_index = {offset: idx for idx, offset in enumerate(self.all_slice_offsets)}
        self.history_slice_indices = [self.slice_offset_to_index[offset] for offset in self.slice_offsets]
        self.target_slice_indices = [self.slice_offset_to_index[offset] for offset in self.target_slice_offsets]
        self.is_lumiere = is_lumiere
        self.registered_data_dir = registered_data_dir.resolve() if registered_data_dir is not None else None
        self.patient_weeks: dict[str, list[int]] = {}
        self.patient_files: dict[str, dict[int, dict[str, Path]]] = {}
        self.slice_cache: dict[tuple[str, int, int | None], torch.Tensor] = {}
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
        import SimpleITK as sitk

        fixed_image = sitk.GetImageFromArray(fixed_vol.astype(np.float32))
        moving_image = sitk.GetImageFromArray(moving_vol.astype(np.float32))

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.20)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=200,
            gradientMagnitudeTolerance=1e-8,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        try:
            final_transform = registration_method.Execute(fixed_image, moving_image)
            resampled = sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID(),
            )
            return sitk.GetArrayFromImage(resampled)
        except Exception as error:
            print(f"Registration failed, returning original volume: {error}")
            return moving_vol

    def _load_week_slices(self, patient_id: str, week: int, reference_week: int | None = None) -> torch.Tensor:
        cache_key = (patient_id, week, reference_week)
        cached = self.slice_cache.get(cache_key)
        if cached is not None:
            return cached

        modality_volumes = {}
        for mod in MODALITIES:
            img_path = self._resolve_modality_path(patient_id, week, mod, reference_week=reference_week)
            volume = nib.load(img_path).get_fdata().astype(np.float32)
            volume = (volume - np.min(volume)) / max(np.max(volume) - np.min(volume), 1e-8)
            modality_volumes[mod] = volume

        if reference_week is not None and reference_week != week and not self._has_registered_volume_set(patient_id, week, reference_week):
            ref_path = self.patient_files[patient_id][reference_week]["CT1"]
            ref_vol = nib.load(ref_path).get_fdata().astype(np.float32)
            ref_vol = (ref_vol - np.min(ref_vol)) / max(np.max(ref_vol) - np.min(ref_vol), 1e-8)
            for mod in MODALITIES:
                modality_volumes[mod] = self._register_to_reference(modality_volumes[mod], ref_vol)

        modality_slices = []
        for mod in MODALITIES:
            volume = modality_volumes[mod]
            height, width = volume.shape[:2]
            height = (height // 16) * 16
            width = (width // 16) * 16
            volume = volume[:height, :width, :]

            center_slice = volume.shape[2] // 2
            selected_slices = []
            for offset in self.all_slice_offsets:
                slice_idx = min(max(center_slice + offset, 0), volume.shape[2] - 1)
                selected_slices.append(torch.from_numpy(volume[:, :, slice_idx]))
            modality_slices.append(torch.stack(selected_slices))

        week_tensor = torch.stack(modality_slices)
        self.slice_cache[cache_key] = week_tensor
        return week_tensor

    def _registered_week_dir(self, patient_id: str, moving_week: int, target_week: int) -> Path | None:
        if self.registered_data_dir is None:
            return None
        return (
            self.registered_data_dir
            / patient_id
            / f"target-{target_week:03d}"
            / f"week-{moving_week:03d}"
            / "DeepBraTumIA-segmentation"
            / "atlas"
            / "skull_strip"
        )

    def _has_registered_volume_set(self, patient_id: str, moving_week: int, target_week: int) -> bool:
        registered_dir = self._registered_week_dir(patient_id, moving_week, target_week)
        if registered_dir is None or not registered_dir.exists():
            return False
        return all((registered_dir / f"{mod.lower()}_skull_strip_registered.nii.gz").exists() for mod in MODALITIES)

    def _resolve_modality_path(self, patient_id: str, week: int, mod: str, reference_week: int | None = None) -> Path:
        if reference_week is not None and reference_week != week and self._has_registered_volume_set(patient_id, week, reference_week):
            registered_dir = self._registered_week_dir(patient_id, week, reference_week)
            assert registered_dir is not None
            return registered_dir / f"{mod.lower()}_skull_strip_registered.nii.gz"
        return self.patient_files[patient_id][week][mod]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        history_week_tensors = []

        for week in sample.history_weeks:
            week_tensor = self._load_week_slices(sample.patient_id, week, reference_week=sample.target_week)
            history_view = week_tensor[:, self.history_slice_indices]
            history_week_tensors.append(history_view.reshape(-1, history_view.shape[2], history_view.shape[3]))

        history_tensor = torch.stack(history_week_tensors, dim=0)

        target_full = self._load_week_slices(sample.patient_id, sample.target_week)
        target_view = target_full[:, self.target_slice_indices]

        latest_history_full = self._load_week_slices(
            sample.patient_id,
            sample.history_weeks[-1],
            reference_week=sample.target_week,
        )
        latest_history_target = latest_history_full[:, self.target_slice_indices]

        return {
            "history": history_tensor,
            "target": target_view.reshape(-1, target_view.shape[2], target_view.shape[3]),
            "latest_history_target": latest_history_target,
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
        for batch_idx in range(batch_size):
            t_span = torch.linspace(
                0.0,
                float(t_delta[batch_idx].item()),
                steps=4,
                dtype=torch.float32,
                device=history.device,
            )
            evolved = odeint(self.ode_func, hidden[batch_idx : batch_idx + 1], t_span, method="midpoint")
            outputs.append(torch.sigmoid(self.decoder(evolved[-1])))
        return torch.cat(outputs, dim=0)


GliomaNeuralODEModel = HistoryConditionedGliomaNeuralODEModel


def build_holdout_last_pair_split(dataset: NeuralODEDataset) -> tuple[list[int], list[int]]:
    latest_target_by_patient = {patient_id: max(weeks) for patient_id, weeks in dataset.patient_weeks.items() if weeks}
    holdout_indices = sorted(
        idx
        for idx, sample in enumerate(dataset.samples)
        if sample.target_week == latest_target_by_patient[sample.patient_id]
    )
    train_indices = [idx for idx in range(len(dataset.samples)) if idx not in holdout_indices]
    return train_indices, holdout_indices


def modality_metric_dict(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metric_fn,
    target_slice_count: int,
) -> dict[str, float]:
    pred_modal = reshape_stacked_modalities(prediction, target_slice_count)
    target_modal = reshape_stacked_modalities(target, target_slice_count)
    if pred_modal.dim() == 5:
        pred_modal = pred_modal[0]
        target_modal = target_modal[0]
    return {
        mod: float(metric_fn(pred_modal[idx], target_modal[idx]).item())
        for idx, mod in enumerate(MODALITIES)
    }


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
            per_modality_mse = modality_metric_dict(prediction, target, mse_crit, len(dataset.target_slice_offsets))
            per_modality_mae = modality_metric_dict(prediction, target, mae_crit, len(dataset.target_slice_offsets))

            pred_flair = reshape_stacked_modalities(prediction, len(dataset.target_slice_offsets))[0, 0].detach().cpu().numpy()
            true_flair = reshape_stacked_modalities(target, len(dataset.target_slice_offsets))[0, 0].detach().cpu().numpy()
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
        baseline_prediction = sample["latest_history_target"].reshape(
            1,
            len(MODALITIES) * len(dataset.target_slice_offsets),
            sample["target"].shape[1],
            sample["target"].shape[2],
        )
        target = sample["target"].unsqueeze(0)
        mse = float(mse_crit(baseline_prediction, target).item())
        mae = float(mae_crit(baseline_prediction, target).item())
        per_modality_mse = modality_metric_dict(
            baseline_prediction,
            target,
            mse_crit,
            len(dataset.target_slice_offsets),
        )
        per_modality_mae = modality_metric_dict(
            baseline_prediction,
            target,
            mae_crit,
            len(dataset.target_slice_offsets),
        )
        pred_flair = reshape_stacked_modalities(baseline_prediction, len(dataset.target_slice_offsets))[0, 0].numpy()
        true_flair = reshape_stacked_modalities(target, len(dataset.target_slice_offsets))[0, 0].numpy()
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
            mod: float(sum(float(row["per_modality_mse"][mod]) for row in patient_rows) / len(patient_rows))
            for mod in MODALITIES
        }
        avg_per_modality_mae = {
            mod: float(sum(float(row["per_modality_mae"][mod]) for row in patient_rows) / len(patient_rows))
            for mod in MODALITIES
        }
        patient_summary[patient_id] = {
            "count": len(patient_rows),
            "avg_mse": float(sum(float(row["mse"]) for row in patient_rows) / len(patient_rows)),
            "avg_mae": float(sum(float(row["mae"]) for row in patient_rows) / len(patient_rows)),
            "avg_relative_flair_volume_diff": float(
                sum(float(row["relative_flair_volume_diff"]) for row in patient_rows) / len(patient_rows)
            ),
            "avg_per_modality_mse": avg_per_modality_mse,
            "avg_per_modality_mae": avg_per_modality_mae,
        }
    avg_per_modality_mse = {
        mod: float(sum(float(row["per_modality_mse"][mod]) for row in rows) / len(rows))
        for mod in MODALITIES
    }
    avg_per_modality_mae = {
        mod: float(sum(float(row["per_modality_mae"][mod]) for row in rows) / len(rows))
        for mod in MODALITIES
    }
    return {
        "count": len(rows),
        "avg_mse": float(sum(float(row["mse"]) for row in rows) / len(rows)),
        "avg_mae": float(sum(float(row["mae"]) for row in rows) / len(rows)),
        "avg_relative_flair_volume_diff": float(
            sum(float(row["relative_flair_volume_diff"]) for row in rows) / len(rows)
        ),
        "avg_per_modality_mse": avg_per_modality_mse,
        "avg_per_modality_mae": avg_per_modality_mae,
        "by_patient": patient_summary,
    }


def flatten_metric_row(row: dict[str, object]) -> dict[str, object]:
    flat = {
        "patient_id": row["patient_id"],
        "history_weeks": ",".join(str(week) for week in row["history_weeks"]),
        "target_week": row["target_week"],
        "mse": row["mse"],
        "mae": row["mae"],
        "relative_flair_volume_diff": row["relative_flair_volume_diff"],
    }
    per_modality_mse = row.get("per_modality_mse") or {}
    per_modality_mae = row.get("per_modality_mae") or {}
    for modality in MODALITIES:
        flat[f"{modality.lower()}_mse"] = per_modality_mse.get(modality)
        flat[f"{modality.lower()}_mae"] = per_modality_mae.get(modality)
    return flat


def write_metric_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "patient_id",
        "history_weeks",
        "target_week",
        "mse",
        "mae",
        "relative_flair_volume_diff",
    ]
    for modality in MODALITIES:
        fieldnames.extend([f"{modality.lower()}_mse", f"{modality.lower()}_mae"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_metric_row(row))


def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if losses:
        ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
        ax.set_ylabel("Loss")
    else:
        ax.text(0.5, 0.5, "Training skipped", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("N/A")
    ax.set_title("Neural ODE Training Loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def add_contour_overlay(axis, image: np.ndarray, color: str) -> None:
    if np.max(image) <= 0:
        return
    level = max(float(np.max(image)) * 0.45, 0.15)
    try:
        axis.contour(image, levels=[level], colors=color, linewidths=0.6)
    except ValueError:
        return


def plot_prediction(
    sample: dict[str, object],
    prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    output_path: Path,
    target_slice_offsets: tuple[int, ...],
    model_mse: float,
    baseline_mse: float,
) -> None:
    target_slice_count = len(target_slice_offsets)
    latest_history = sample["latest_history_target"].detach().cpu()
    target = sample["target"].detach().cpu()
    prediction = prediction[0].detach().cpu()
    baseline_prediction = baseline_prediction[0].detach().cpu()

    if latest_history.dim() == 4:
        latest_history = latest_history.reshape(-1, latest_history.shape[2], latest_history.shape[3])
    latest_history = reshape_stacked_modalities(latest_history, target_slice_count)
    target = reshape_stacked_modalities(target, target_slice_count)
    prediction = reshape_stacked_modalities(prediction, target_slice_count)
    baseline_prediction = reshape_stacked_modalities(baseline_prediction, target_slice_count)

    rows = len(MODALITIES) * target_slice_count
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.2 * rows))
    axes = np.atleast_2d(axes)

    for mod_idx, mod in enumerate(MODALITIES):
        for slice_idx, offset in enumerate(target_slice_offsets):
            row = mod_idx * target_slice_count + slice_idx
            history_img = latest_history[mod_idx, slice_idx].numpy()
            pred_img = prediction[mod_idx, slice_idx].numpy()
            target_img = target[mod_idx, slice_idx].numpy()
            error_img = np.abs(pred_img - target_img)
            baseline_img = baseline_prediction[mod_idx, slice_idx].numpy()

            views = [
                (history_img, "gray", f"{mod} history ({offset:+d})"),
                (pred_img, "gray", f"{mod} prediction ({offset:+d})"),
                (target_img, "gray", f"{mod} target ({offset:+d})"),
                (error_img, "inferno", f"{mod} abs error ({offset:+d})"),
            ]

            for col, (image, cmap, title) in enumerate(views):
                axis = axes[row, col]
                vmax = 1.0 if col < 3 else max(float(error_img.max()), 1e-3)
                axis.imshow(image, cmap=cmap, vmin=0.0, vmax=vmax)
                if col == 0 and mod in {"FLAIR", "CT1"}:
                    add_contour_overlay(axis, target_img, "lime")
                if col == 1 and mod in {"FLAIR", "CT1"}:
                    add_contour_overlay(axis, pred_img, "cyan")
                if col == 2 and mod in {"FLAIR", "CT1"}:
                    add_contour_overlay(axis, target_img, "lime")
                axis.set_title(title, fontsize=9)
                axis.axis("off")

                if col == 3:
                    axis.text(
                        0.02,
                        0.98,
                        f"baseline MSE {F.mse_loss(torch.from_numpy(baseline_img), torch.from_numpy(target_img)).item():.4f}",
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
                    )

    history_weeks = sample["history_weeks"].tolist()
    target_week = int(sample["target_week"])
    improvement = (baseline_mse - model_mse) / max(baseline_mse, 1e-8)
    fig.suptitle(
        (
            f"History weeks {history_weeks} -> target week {target_week} | "
            f"model MSE={model_mse:.4f} | baseline MSE={baseline_mse:.4f} | "
            f"improvement={improvement:+.1%}"
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=180)
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
            pad = torch.zeros(pad_len, history_channels, height, width, dtype=history.dtype)
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
    target_slice_count: int,
) -> tuple[list[float], list[dict[str, float]]]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_history_batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_losses = []
    epoch_details = []

    for epoch in range(epochs):
        model.train()
        running = {"total_loss": 0.0, "mse_loss": 0.0, "l1_loss": 0.0, "edge_loss": 0.0, "focus_l1_loss": 0.0}
        for batch in dataloader:
            history = batch["history"].to(device)
            target = batch["target"].to(device)
            time_delta = batch["time_delta"].to(device)
            history_weeks = batch["history_weeks"].to(device)
            history_mask = batch["history_mask"].to(device)

            optimizer.zero_grad()
            prediction = model(history, time_delta, history_weeks, history_mask)
            loss_dict = compute_losses(prediction, target, target_slice_count)
            loss_dict["total_loss"].backward()
            optimizer.step()

            for key in running:
                running[key] += float(loss_dict[key].item())

        num_batches = len(dataloader)
        averaged = {key: value / num_batches for key, value in running.items()}
        epoch_losses.append(averaged["total_loss"])
        epoch_details.append({"epoch": epoch + 1, **averaged})
        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"loss={averaged['total_loss']:.6f} | "
            f"mse={averaged['mse_loss']:.6f} | "
            f"edge={averaged['edge_loss']:.6f}"
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


def resolve_patient_dirs(search_root: Path, patient_names: list[str]) -> list[Path]:
    candidate_roots = [search_root.resolve(), (search_root / "Imaging").resolve()]
    resolved_dirs: list[Path] = []
    missing_names: list[str] = []

    for patient_name in patient_names:
        resolved = None
        for root in candidate_roots:
            candidate = root / patient_name
            if candidate.is_dir():
                resolved = candidate
                break
        if resolved is None:
            missing_names.append(patient_name)
        else:
            resolved_dirs.append(resolved)

    if missing_names:
        searched_roots = ", ".join(str(root) for root in candidate_roots)
        raise FileNotFoundError(
            f"Missing patient directories for {missing_names}. Searched under: {searched_roots}"
        )

    return resolved_dirs


def run_experiment(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    patient_names: list[str],
    run_name: str,
    data_dir: Path | None = None,
    is_lumiere: bool = False,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    epochs_override: int | None = None,
    stage_name: str = "experiment",
    pretrained_from: str | None = None,
) -> dict[str, object]:
    search_root = data_dir or repo_root
    patient_dirs = resolve_patient_dirs(search_root, patient_names)

    device = get_device(args.device)
    run_dir = repo_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Patients: {patient_names}")
    print(f"Stage: {stage_name}")

    dataset = NeuralODEDataset(
        patient_dirs=patient_dirs,
        history_mode=args.history_mode,
        context_size=args.context_size,
        slice_offsets=args.slice_offsets,
        target_slice_offsets=args.target_slice_offsets,
        is_lumiere=is_lumiere,
        registered_data_dir=args.registered_data_dir,
    )
    print(f"History-conditioned samples: {len(dataset)}")
    print(f"Patient weeks: {dataset.patient_weeks}")

    train_indices = list(range(len(dataset.samples)))
    holdout_indices: list[int] = []
    if args.holdout_last_pair:
        train_indices, holdout_indices = build_holdout_last_pair_split(dataset)
        print(f"Train samples after holdout: {len(train_indices)}")
        print(f"Holdout samples: {len(holdout_indices)}")

    sample = dataset[0]
    model = build_model(
        week_input_channels=int(sample["history"].shape[1]),
        output_channels=int(sample["target"].shape[0]),
        max_weeks=max(max(weeks) for weeks in dataset.patient_weeks.values()),
        model_size=args.model_size,
    ).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)

    epoch_count = epochs_override if epochs_override is not None else args.epochs
    train_skipped = False
    if not train_indices:
        if initial_state_dict is None:
            raise RuntimeError(
                "Holdout split consumed all history-conditioned training samples. "
                "Add more longitudinal weeks for the selected patients or rerun with "
                "--no-holdout-last-pair."
            )
        print("No train samples left after holdout; using pretrained weights without fine-tuning.")
        losses: list[float] = []
        epoch_details: list[dict[str, float]] = []
        train_skipped = True
    else:
        train_dataset: Dataset = Subset(dataset, train_indices)
        losses, epoch_details = train_model(
            model=model,
            dataset=train_dataset,
            device=device,
            epochs=epoch_count,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            target_slice_count=len(dataset.target_slice_offsets),
        )

    checkpoint_path = run_dir / f"attention_unet_neural_ode_{args.model_size}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    plot_loss_curve(losses, run_dir / "loss_curve.png")

    train_metrics = evaluate_model(model, dataset, device, indices=train_indices) if train_indices else []
    holdout_metrics = evaluate_model(model, dataset, device, indices=holdout_indices) if holdout_indices else []
    all_metrics = evaluate_model(model, dataset, device)
    baseline_train_metrics = evaluate_persistence_baseline(dataset, indices=train_indices) if train_indices else []
    baseline_holdout_metrics = (
        evaluate_persistence_baseline(dataset, indices=holdout_indices) if holdout_indices else []
    )
    baseline_all_metrics = evaluate_persistence_baseline(dataset)
    metric_csv_paths = {
        "train_history_metrics": run_dir / "train_history_metrics.csv",
        "holdout_history_metrics": run_dir / "holdout_history_metrics.csv",
        "all_history_metrics": run_dir / "all_history_metrics.csv",
        "baseline_train_history_metrics": run_dir / "baseline_train_history_metrics.csv",
        "baseline_holdout_history_metrics": run_dir / "baseline_holdout_history_metrics.csv",
        "baseline_all_history_metrics": run_dir / "baseline_all_history_metrics.csv",
    }
    write_metric_rows(train_metrics, metric_csv_paths["train_history_metrics"])
    write_metric_rows(holdout_metrics, metric_csv_paths["holdout_history_metrics"])
    write_metric_rows(all_metrics, metric_csv_paths["all_history_metrics"])
    write_metric_rows(baseline_train_metrics, metric_csv_paths["baseline_train_history_metrics"])
    write_metric_rows(baseline_holdout_metrics, metric_csv_paths["baseline_holdout_history_metrics"])
    write_metric_rows(baseline_all_metrics, metric_csv_paths["baseline_all_history_metrics"])

    print(f"Model holdout summary: {summarize_metric_rows(holdout_metrics)}")
    print(f"Baseline holdout summary: {summarize_metric_rows(baseline_holdout_metrics)}")

    summary_predictions = []
    mse_crit = nn.MSELoss()
    with torch.no_grad():
        prediction_indices = holdout_indices or [len(dataset.samples) - 1]
        for idx in prediction_indices:
            sample = dataset[idx]
            patient_id = str(sample["patient_id"])
            patient_run_dir = run_dir / patient_id
            patient_run_dir.mkdir(parents=True, exist_ok=True)
            history = sample["history"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            time_delta = sample["time_delta"].unsqueeze(0).to(device)
            history_weeks = sample["history_weeks"].unsqueeze(0).to(device)
            history_mask = torch.ones((1, history.shape[1]), dtype=torch.bool, device=device)
            prediction = model(history, time_delta, history_weeks, history_mask)
            baseline_prediction = sample["latest_history_target"].reshape(
                1,
                len(MODALITIES) * len(dataset.target_slice_offsets),
                sample["target"].shape[1],
                sample["target"].shape[2],
            )
            model_mse = float(mse_crit(prediction, target).item())
            baseline_mse = float(mse_crit(baseline_prediction.to(device), target).item())
            output_path = patient_run_dir / f"prediction_to_week_{int(sample['target_week'])}.png"
            plot_prediction(
                sample=sample,
                prediction=prediction,
                baseline_prediction=baseline_prediction,
                output_path=output_path,
                target_slice_offsets=dataset.target_slice_offsets,
                model_mse=model_mse,
                baseline_mse=baseline_mse,
            )
            summary_predictions.append(
                {
                    "patient_id": patient_id,
                    "history_weeks": sample["history_weeks"].tolist(),
                    "target_week": int(sample["target_week"]),
                    "visualization": str(output_path),
                    "model_mse": model_mse,
                    "baseline_mse": baseline_mse,
                    "relative_improvement": float((baseline_mse - model_mse) / max(baseline_mse, 1e-8)),
                }
            )

    metadata = {
        "run_name": run_name,
        "stage_name": stage_name,
        "pretrained_from": pretrained_from,
        "repo_root": str(repo_root),
        "device": str(device),
        "requested_device": args.device,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "patients": patient_names,
        "data_dir": str(search_root),
        "registered_data_dir": str(args.registered_data_dir.resolve()) if args.registered_data_dir is not None else None,
        "patient_weeks": dataset.patient_weeks,
        "history_mode": args.history_mode,
        "context_size": args.context_size,
        "slice_offsets": list(dataset.slice_offsets),
        "target_slice_offsets": list(dataset.target_slice_offsets),
        "num_samples": len(dataset),
        "train_sample_count": len(train_indices),
        "holdout_sample_count": len(holdout_indices),
        "epochs": epoch_count,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_size": args.model_size,
        "holdout_last_pair": args.holdout_last_pair,
        "train_skipped": train_skipped,
        "cohort_pretrain_epochs": args.cohort_pretrain_epochs,
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
        "baseline_name": "latest_history_registered_target_slice_stack",
        "baseline_train_history_metrics": baseline_train_metrics,
        "baseline_holdout_history_metrics": baseline_holdout_metrics,
        "baseline_all_history_metrics": baseline_all_metrics,
        "baseline_train_metric_summary": summarize_metric_rows(baseline_train_metrics),
        "baseline_holdout_metric_summary": summarize_metric_rows(baseline_holdout_metrics),
        "baseline_all_metric_summary": summarize_metric_rows(baseline_all_metrics),
        "metric_csvs": {key: str(path) for key, path in metric_csv_paths.items()},
        "prediction_visualizations": summary_predictions,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved summary: {summary_path}")
    return {"run_dir": run_dir, "checkpoint_path": checkpoint_path, "summary_path": summary_path}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    repo_root = args.repo_root.resolve()

    if args.target_slice_offsets is None:
        args.target_slice_offsets = list(args.slice_offsets)

    if args.registered_data_dir is not None:
        args.registered_data_dir = args.registered_data_dir.resolve()
        if not args.registered_data_dir.is_dir():
            raise FileNotFoundError(f"Registered data directory not found: {args.registered_data_dir}")

    search_root = args.data_dir.resolve() if args.data_dir else repo_root
    if args.patients:
        patient_names = args.patients
    elif args.lumiere:
        patient_root = search_root if any(search_root.glob("Patient-*")) else search_root / "Imaging"
        patient_names = sorted(path.name for path in patient_root.glob("Patient-*") if path.is_dir())
    else:
        patient_names = sorted(path.name for path in search_root.glob("patient_*") if path.is_dir())

    if not patient_names:
        raise FileNotFoundError(f"No patient directories were found under {search_root}")

    if args.separate_patient_runs:
        base_run_name = args.run_name or datetime.now().strftime("neural_ode_run_%Y%m%d_%H%M%S")
        initial_state_dict = None
        if args.cohort_pretrain_epochs > 0 and len(patient_names) > 1:
            print("\n=== Cohort pretraining stage ===")
            pretrain_result = run_experiment(
                args=args,
                repo_root=repo_root,
                patient_names=patient_names,
                run_name=f"{base_run_name}_cohort_pretrain",
                data_dir=args.data_dir,
                is_lumiere=args.lumiere,
                epochs_override=args.cohort_pretrain_epochs,
                stage_name="cohort_pretrain",
            )
            initial_state_dict = torch.load(pretrain_result["checkpoint_path"], map_location="cpu")

        run_dirs = []
        for patient_name in patient_names:
            print(f"\n=== Separate Neural ODE run for {patient_name} ===")
            try:
                result = run_experiment(
                    args=args,
                    repo_root=repo_root,
                    patient_names=[patient_name],
                    run_name=f"{base_run_name}_{patient_name}",
                    data_dir=args.data_dir,
                    is_lumiere=args.lumiere,
                    initial_state_dict=copy.deepcopy(initial_state_dict) if initial_state_dict is not None else None,
                    stage_name="patient_finetune" if initial_state_dict is not None else "patient_only",
                    pretrained_from=(
                        str((repo_root / "runs" / f"{base_run_name}_cohort_pretrain" / f"attention_unet_neural_ode_{args.model_size}.pt"))
                        if initial_state_dict is not None
                        else None
                    ),
                )
                run_dirs.append(result["run_dir"])
            except (RuntimeError, FileNotFoundError) as error:
                print(f"Skipping {patient_name} due to error: {error}")
                continue
        print("\nCompleted Neural ODE runs:")
        for run_dir in run_dirs:
            print(run_dir)
        if not run_dirs:
            raise RuntimeError("No separate patient runs completed successfully.")
        return

    run_name = args.run_name or datetime.now().strftime("neural_ode_run_%Y%m%d_%H%M%S")
    run_experiment(
        args=args,
        repo_root=repo_root,
        patient_names=patient_names,
        run_name=run_name,
        data_dir=args.data_dir,
        is_lumiere=args.lumiere,
        stage_name="cohort_run",
    )


if __name__ == "__main__":
    main()

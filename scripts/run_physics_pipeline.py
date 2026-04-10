from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


MODALITIES = ("FLAIR", "T1", "T2", "CT1")
WEEK_PATTERN = re.compile(r"wk(\d+)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train and run the 3D physics-informed glioma model on local patient folders."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root that contains patient_* folders.",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Specific patient folder names. Defaults to all patient_* folders under repo root.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--resize-dim",
        nargs=3,
        type=int,
        default=(48, 48, 48),
        metavar=("D", "H", "W"),
        help="Resize MRI volumes before training.",
    )
    parser.add_argument(
        "--forecast-weeks",
        type=float,
        nargs=2,
        default=(12.0, 24.0),
        metavar=("SHORT", "LONG"),
        help="Forecast horizons in weeks.",
    )
    parser.add_argument(
        "--therapy-weeks",
        type=float,
        default=24.0,
        help="Therapy simulation horizon in weeks.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
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
class TrainingPair:
    patient_id: str
    week_t0: int
    week_t1: int
    dt_years: float


class MultiPatientGliomaDataset(Dataset):
    def __init__(
        self,
        patient_dirs: Iterable[Path],
        modalities: tuple[str, ...] = MODALITIES,
        resize_dim: tuple[int, int, int] = (48, 48, 48),
    ) -> None:
        self.patient_dirs = [path.resolve() for path in patient_dirs]
        self.patient_dirs_by_name = {path.name: path.resolve() for path in patient_dirs}
        self.modalities = modalities
        self.resize_dim = tuple(resize_dim)
        self.patient_weeks: dict[str, list[int]] = {}
        self.patient_files: dict[str, dict[int, dict[str, Path]]] = {}
        self.pairs: list[TrainingPair] = []
        self.volume_cache: dict[tuple[str, int], torch.Tensor] = {}

        for patient_dir in self.patient_dirs:
            weeks = self._discover_weeks(patient_dir)
            self.patient_weeks[patient_dir.name] = weeks
            for week_t0, week_t1 in zip(weeks, weeks[1:]):
                self.pairs.append(
                    TrainingPair(
                        patient_id=patient_dir.name,
                        week_t0=week_t0,
                        week_t1=week_t1,
                        dt_years=(week_t1 - week_t0) / 52.0,
                    )
                )

        if not self.pairs:
            raise RuntimeError("No valid patient scan pairs were found.")

    def _discover_weeks(self, patient_dir: Path) -> list[int]:
        week_to_paths: dict[int, dict[str, Path]] = {}
        for mod in self.modalities:
            for mod_file in sorted(patient_dir.glob(f"{mod}_wk*.nii")):
                week = extract_week(mod_file.name)
                week_to_paths.setdefault(week, {})[mod] = mod_file

        valid_week_map = {
            week: paths
            for week, paths in week_to_paths.items()
            if all(mod in paths for mod in self.modalities)
        }
        weeks = sorted(valid_week_map)
        self.patient_files[patient_dir.name] = valid_week_map
        if len(weeks) < 2:
            raise RuntimeError(f"Patient {patient_dir.name} has fewer than two valid timepoints.")
        return weeks

    def _load_volume(self, patient_id: str, week: int) -> torch.Tensor:
        cache_key = (patient_id, week)
        cached = self.volume_cache.get(cache_key)
        if cached is not None:
            return cached

        channels = []
        for mod in self.modalities:
            img_path = self.patient_files[patient_id][week][mod]
            img = nib.load(img_path).get_fdata().astype(np.float32)
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            channels.append(torch.from_numpy(img))

        volume = torch.stack(channels).unsqueeze(0)
        volume = F.interpolate(
            volume,
            size=self.resize_dim,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        self.volume_cache[cache_key] = volume
        return volume

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        pair = self.pairs[idx]
        x_t0 = self._load_volume(pair.patient_id, pair.week_t0)
        y_t1 = self._load_volume(pair.patient_id, pair.week_t1)[0:1]
        return {
            "x_t0": x_t0,
            "y_t1": y_t1,
            "dt": torch.tensor(pair.dt_years, dtype=torch.float32),
            "patient_id": pair.patient_id,
            "weeks": (pair.week_t0, pair.week_t1),
        }


class ParameterEstimator3D(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.to_params = nn.Conv3d(32, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.enc(x)
        raw_params = self.to_params(features)
        diffusion = torch.sigmoid(raw_params[:, 0:1]) * 0.5
        proliferation = torch.sigmoid(raw_params[:, 1:2]) * 0.5
        return diffusion, proliferation


class ReactionDiffusionSolver(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        kernel = torch.zeros(1, 1, 3, 3, 3)
        kernel[0, 0, 1, 1, 1] = -6.0
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        self.register_buffer("laplacian_kernel", kernel.to(device))

    def forward(
        self,
        c_init: torch.Tensor,
        diffusion_map: torch.Tensor,
        proliferation_map: torch.Tensor,
        dt: torch.Tensor | float,
        steps: int = 24,
        therapy_map: torch.Tensor | None = None,
        therapy_strength: float = 2.0,
    ) -> torch.Tensor:
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(float(dt), dtype=c_init.dtype, device=c_init.device)
        delta_t = dt / steps
        concentration = c_init

        for _ in range(steps):
            laplacian_c = F.conv3d(concentration, self.laplacian_kernel, padding=1)
            diffusion = diffusion_map * laplacian_c
            reaction = proliferation_map * concentration * (1.0 - concentration)
            kill = 0.0
            if therapy_map is not None:
                kill = therapy_map * concentration * therapy_strength
            concentration = concentration + (diffusion + reaction - kill) * delta_t
            concentration = torch.clamp(concentration, 0.0, 1.0)

        return concentration


class GliomaPhysicsModel(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.estimator = ParameterEstimator3D(in_channels=len(MODALITIES))
        self.solver = ReactionDiffusionSolver(device)

    def forward(self, x_t0: torch.Tensor, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diffusion_map, proliferation_map = self.estimator(x_t0)
        c_t0 = x_t0[:, 0:1]
        c_t1_pred = self.solver(c_t0, diffusion_map, proliferation_map, dt=dt.mean(), steps=24)
        return c_t1_pred, diffusion_map, proliferation_map


def smoothness_loss_3d(volume: torch.Tensor) -> torch.Tensor:
    losses = [
        torch.mean(torch.abs(volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :])),
        torch.mean(torch.abs(volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :])),
        torch.mean(torch.abs(volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1])),
    ]
    return sum(losses) / len(losses)


def train_model(
    model: GliomaPhysicsModel,
    dataset: MultiPatientGliomaDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[list[float], list[dict[str, float]]]:
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_crit = nn.MSELoss()

    epoch_losses: list[float] = []
    epoch_details: list[dict[str, float]] = []

    for epoch in range(epochs):
        running_total = 0.0
        running_fit = 0.0
        running_smooth = 0.0
        running_variance = 0.0

        for batch in dataloader:
            x_t0 = batch["x_t0"].to(device)
            y_t1 = batch["y_t1"].to(device)
            dt = batch["dt"].to(device)

            optimizer.zero_grad()
            pred_t1, diffusion_map, proliferation_map = model(x_t0, dt)

            fit_loss = mse_crit(pred_t1, y_t1)
            smooth_loss = smoothness_loss_3d(diffusion_map) + smoothness_loss_3d(proliferation_map)
            map_std = torch.std(diffusion_map)
            variance_penalty = torch.relu(torch.tensor(0.05, device=device) - map_std)

            total_loss = fit_loss + (0.001 * smooth_loss) + (10.0 * variance_penalty)
            total_loss.backward()
            optimizer.step()

            running_total += float(total_loss.item())
            running_fit += float(fit_loss.item())
            running_smooth += float(smooth_loss.item())
            running_variance += float(variance_penalty.item())

        num_batches = len(dataloader)
        average_total = running_total / num_batches
        epoch_losses.append(average_total)
        epoch_details.append(
            {
                "epoch": epoch + 1,
                "total_loss": average_total,
                "fit_loss": running_fit / num_batches,
                "smooth_loss": running_smooth / num_batches,
                "variance_penalty": running_variance / num_batches,
                "d_min": float(diffusion_map.min().item()),
                "d_max": float(diffusion_map.max().item()),
            }
        )
        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"loss={average_total:.6f} | "
            f"fit={running_fit / num_batches:.6f} | "
            f"D=[{diffusion_map.min():.3f}, {diffusion_map.max():.3f}]"
        )

    return epoch_losses, epoch_details


def evaluate_one_step(
    model: GliomaPhysicsModel,
    dataset: MultiPatientGliomaDataset,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    metrics = []
    mse_crit = nn.MSELoss()
    with torch.no_grad():
        for pair in dataset.pairs:
            sample = dataset[dataset.pairs.index(pair)]
            x_t0 = sample["x_t0"].unsqueeze(0).to(device)
            y_t1 = sample["y_t1"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device)
            pred_t1, _, _ = model(x_t0, dt)
            mse = float(mse_crit(pred_t1, y_t1).item())
            pred_np = pred_t1[0, 0].detach().cpu().numpy()
            true_np = y_t1[0, 0].detach().cpu().numpy()
            pred_volume = float((pred_np > 0.5).sum())
            true_volume = float((true_np > 0.5).sum())
            vol_diff = abs(pred_volume - true_volume) / max(true_volume, 1.0)
            metrics.append(
                {
                    "patient_id": pair.patient_id,
                    "week_t0": pair.week_t0,
                    "week_t1": pair.week_t1,
                    "mse": mse,
                    "relative_volume_diff": vol_diff,
                }
            )
    return metrics


def detect_tumor_center(concentration: torch.Tensor) -> tuple[int, int, int]:
    volume = concentration[0, 0].detach().cpu().numpy()
    max_index = np.unravel_index(np.argmax(volume), volume.shape)
    return int(max_index[0]), int(max_index[1]), int(max_index[2])


def create_radiation_plan(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    radius: int,
) -> torch.Tensor:
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2)
    rad_map = np.zeros(shape, dtype=np.float32)
    rad_map[dist <= radius] = 1.0
    return torch.from_numpy(rad_map).unsqueeze(0).unsqueeze(0)


def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_forecast(
    patient_id: str,
    last_week: int,
    current: torch.Tensor,
    short_term: torch.Tensor,
    long_term: torch.Tensor,
    diffusion_map: torch.Tensor,
    output_path: Path,
    forecast_weeks: tuple[float, float],
) -> None:
    mid = current.shape[2] // 2
    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    ax[0].imshow(current[0, 0, mid].detach().cpu(), cmap="gray")
    ax[0].set_title(f"{patient_id} Current\nWeek {last_week}")
    ax[1].imshow(short_term[0, 0, mid].detach().cpu(), cmap="magma", vmin=0, vmax=1)
    ax[1].set_title(f"Forecast +{int(forecast_weeks[0])} Weeks")
    ax[2].imshow(long_term[0, 0, mid].detach().cpu(), cmap="magma", vmin=0, vmax=1)
    ax[2].set_title(f"Forecast +{int(forecast_weeks[1])} Weeks")
    im = ax[3].imshow(diffusion_map[0, 0, mid].detach().cpu(), cmap="jet")
    ax[3].set_title("Inferred Diffusion")
    fig.colorbar(im, ax=ax[3])
    fig.suptitle(f"{patient_id}: Physics-Informed Forecast", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_therapy(
    patient_id: str,
    last_week: int,
    current: torch.Tensor,
    natural: torch.Tensor,
    treated: torch.Tensor,
    therapy_map: torch.Tensor,
    output_path: Path,
    therapy_weeks: float,
) -> None:
    mid = current.shape[2] // 2
    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    ax[0].imshow(current[0, 0, mid].detach().cpu(), cmap="gray")
    ax[0].set_title(f"Start\nWeek {last_week}")
    ax[1].imshow(natural[0, 0, mid].detach().cpu(), cmap="magma", vmin=0, vmax=1)
    ax[1].set_title(f"Natural +{int(therapy_weeks)} Weeks")
    ax[2].imshow(therapy_map[0, 0, mid].detach().cpu(), cmap="winter", alpha=0.5)
    ax[2].imshow(treated[0, 0, mid].detach().cpu(), cmap="magma", alpha=0.75, vmin=0, vmax=1)
    ax[2].set_title("With Radiation")
    diff = (natural - treated)[0, 0, mid].detach().cpu()
    ax[3].imshow(diff, cmap="inferno")
    ax[3].set_title("Tumor Reduction")
    fig.suptitle(f"{patient_id}: Therapy Simulation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def serialize_patient_weeks(patient_weeks: dict[str, list[int]]) -> dict[str, list[int]]:
    return {patient_id: list(weeks) for patient_id, weeks in patient_weeks.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    repo_root = args.repo_root.resolve()
    patient_names = args.patients or sorted(path.name for path in repo_root.glob("patient_*") if path.is_dir())
    patient_dirs = [repo_root / name for name in patient_names]
    missing = [path for path in patient_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing patient directories: {missing}")

    device = get_device()
    resize_dim = tuple(args.resize_dim)
    run_name = args.run_name or datetime.now().strftime("physics_run_%Y%m%d_%H%M%S")
    run_dir = repo_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Patients: {patient_names}")

    dataset = MultiPatientGliomaDataset(patient_dirs=patient_dirs, resize_dim=resize_dim)
    print(f"Training pairs: {len(dataset)}")
    print(f"Patient weeks: {serialize_patient_weeks(dataset.patient_weeks)}")

    model = GliomaPhysicsModel(device).to(device)
    losses, epoch_details = train_model(
        model=model,
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    checkpoint_path = run_dir / "physics_glioma_model_dual_patient.pt"
    torch.save(model.state_dict(), checkpoint_path)
    plot_loss_curve(losses, run_dir / "loss_curve.png")

    one_step_metrics = evaluate_one_step(model, dataset, device)

    with torch.no_grad():
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            last_week = dataset.patient_weeks[patient_id][-1]
            patient_run_dir = run_dir / patient_id
            patient_run_dir.mkdir(parents=True, exist_ok=True)

            x_t0 = dataset._load_volume(patient_id, last_week).unsqueeze(0).to(device)
            diffusion_map, proliferation_map = model.estimator(x_t0)
            c_current = x_t0[:, 0:1]

            short_dt = args.forecast_weeks[0] / 52.0
            long_dt = args.forecast_weeks[1] / 52.0
            c_short = model.solver(c_current, diffusion_map, proliferation_map, dt=short_dt, steps=32)
            c_long = model.solver(
                c_short,
                diffusion_map,
                proliferation_map,
                dt=(long_dt - short_dt),
                steps=32,
            )

            tumor_center = detect_tumor_center(c_current)
            therapy_map = create_radiation_plan(
                shape=resize_dim,
                center=tumor_center,
                radius=max(4, resize_dim[0] // 6),
            ).to(device)
            natural = model.solver(
                c_current,
                diffusion_map,
                proliferation_map,
                dt=args.therapy_weeks / 52.0,
                steps=36,
            )
            treated = model.solver(
                c_current,
                diffusion_map,
                proliferation_map,
                dt=args.therapy_weeks / 52.0,
                steps=36,
                therapy_map=therapy_map,
            )

            plot_forecast(
                patient_id=patient_id,
                last_week=last_week,
                current=c_current,
                short_term=c_short,
                long_term=c_long,
                diffusion_map=diffusion_map,
                output_path=patient_run_dir / "forecast.png",
                forecast_weeks=tuple(args.forecast_weeks),
            )
            plot_therapy(
                patient_id=patient_id,
                last_week=last_week,
                current=c_current,
                natural=natural,
                treated=treated,
                therapy_map=therapy_map,
                output_path=patient_run_dir / "therapy.png",
                therapy_weeks=args.therapy_weeks,
            )

    metadata = {
        "run_name": run_name,
        "repo_root": str(repo_root),
        "device": str(device),
        "patients": patient_names,
        "patient_weeks": serialize_patient_weeks(dataset.patient_weeks),
        "num_pairs": len(dataset),
        "resize_dim": list(resize_dim),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "forecast_weeks": list(args.forecast_weeks),
        "therapy_weeks": args.therapy_weeks,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "epoch_details": epoch_details,
        "one_step_metrics": one_step_metrics,
    }
    metadata_path = run_dir / "run_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved summary: {metadata_path}")


if __name__ == "__main__":
    main()

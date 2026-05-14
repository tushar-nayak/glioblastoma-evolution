from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ("flair", "t1", "t2", "ct1")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic LUMIERE-style patient folder for smoke testing."
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "test_data")
    parser.add_argument("--patient-id", type=str, default="Patient-999")
    parser.add_argument("--num-weeks", type=int, default=3, help="Number of synthetic week folders to create.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shape", nargs=3, type=int, default=(64, 64, 64))
    return parser.parse_args()


def generate_dummy_nifti(path: Path, rng: np.random.Generator, shape: tuple[int, int, int], week_idx: int) -> None:
    data = rng.random(shape, dtype=np.float32) * 0.15
    center = np.array([dim // 2 for dim in shape], dtype=np.float32)
    center[0] += week_idx
    radius = 8 + week_idx
    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2) <= radius**2
    data[mask] += 0.75
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, path)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    if args.num_weeks < 2:
        raise ValueError("--num-weeks must be at least 2")

    base_dir = args.output_dir.resolve() / args.patient_id
    rng = np.random.default_rng(args.seed)
    shape = tuple(int(dim) for dim in args.shape)

    for week_idx in range(1, args.num_weeks + 1):
        week = f"{week_idx:03d}"
        skull_strip_path = base_dir / f"week-{week}" / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
        skull_strip_path.mkdir(parents=True, exist_ok=True)
        for mod in MODALITIES:
            file_path = skull_strip_path / f"{mod}_skull_strip.nii.gz"
            generate_dummy_nifti(file_path, rng=rng, shape=shape, week_idx=week_idx - 1)


if __name__ == "__main__":
    main()

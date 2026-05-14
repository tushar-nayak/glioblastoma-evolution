from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ("FLAIR", "T1", "T2", "CT1")
MODALITY_FILE_PREFIX = {"FLAIR": "flair", "T1": "t1", "T2": "t2", "CT1": "ct1"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Pre-register LUMIERE volumes to each target week and save full aligned NIfTI files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "lumiere" / "Imaging",
        help="Path to the LUMIERE imaging root or its parent Imaging directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "lumiere_registered",
        help="Directory where registered full-volume NIfTI files will be written.",
    )
    parser.add_argument("--patients", nargs="*", default=None, help="Optional subset of Patient-XXX identifiers.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite registered files that already exist.",
    )
    return parser.parse_args()


def resolve_patient_root(data_dir: Path) -> Path:
    data_dir = data_dir.resolve()
    if any(data_dir.glob("Patient-*")):
        return data_dir
    nested = data_dir / "Imaging"
    if any(nested.glob("Patient-*")):
        return nested
    raise FileNotFoundError(f"Could not find Patient-* directories under {data_dir}")


def discover_patient_week_map(patient_dir: Path) -> dict[int, dict[str, Path]]:
    week_to_paths: dict[int, dict[str, Path]] = {}
    for week_dir in patient_dir.glob("week-*"):
        try:
            week_num = int(week_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue

        skull_strip_path = week_dir / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
        if not skull_strip_path.exists():
            continue

        paths = {}
        for mod, file_prefix in MODALITY_FILE_PREFIX.items():
            matches = list(skull_strip_path.glob(f"{file_prefix}_skull_strip.nii*"))
            if matches:
                paths[mod] = matches[0]
        if all(mod in paths for mod in MODALITIES):
            week_to_paths[week_num] = paths
    return week_to_paths


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    return (volume - np.min(volume)) / max(np.max(volume) - np.min(volume), 1e-8)


def register_to_reference(moving_vol: np.ndarray, fixed_vol: np.ndarray) -> np.ndarray:
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


def registered_week_dir(output_dir: Path, patient_id: str, target_week: int, moving_week: int) -> Path:
    return (
        output_dir
        / patient_id
        / f"target-{target_week:03d}"
        / f"week-{moving_week:03d}"
        / "DeepBraTumIA-segmentation"
        / "atlas"
        / "skull_strip"
    )


def all_outputs_exist(output_dir: Path, patient_id: str, target_week: int, moving_week: int) -> bool:
    week_dir = registered_week_dir(output_dir, patient_id, target_week, moving_week)
    return all((week_dir / f"{MODALITY_FILE_PREFIX[mod]}_skull_strip_registered.nii.gz").exists() for mod in MODALITIES)


def main() -> None:
    args = parse_args()
    patient_root = resolve_patient_root(args.data_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.patients:
        patient_names = args.patients
    else:
        patient_names = sorted(path.name for path in patient_root.glob("Patient-*") if path.is_dir())

    if not patient_names:
        raise FileNotFoundError(f"No patient directories were found under {patient_root}")

    missing_patients = [name for name in patient_names if not (patient_root / name).is_dir()]
    if missing_patients:
        raise FileNotFoundError(f"Missing requested patient directories: {missing_patients}")

    manifest: dict[str, object] = {
        "patient_root": str(patient_root),
        "output_dir": str(output_dir),
        "patients": patient_names,
        "jobs": [],
    }

    for patient_name in patient_names:
        patient_dir = patient_root / patient_name
        week_map = discover_patient_week_map(patient_dir)
        weeks = sorted(week_map)
        if len(weeks) < 2:
            print(f"Skipping {patient_name}: only {len(weeks)} usable weeks.")
            continue

        print(f"\n=== {patient_name} ===")
        for target_idx in range(1, len(weeks)):
            target_week = weeks[target_idx]
            fixed_ct1_path = week_map[target_week]["CT1"]
            fixed_ct1_img = nib.load(fixed_ct1_path)
            fixed_ct1 = normalize_volume(fixed_ct1_img.get_fdata())
            print(f"Target week {target_week:03d}: registering {target_idx} prior weeks")

            for moving_week in weeks[:target_idx]:
                if not args.overwrite and all_outputs_exist(output_dir, patient_name, target_week, moving_week):
                    print(f"  Reusing cache for week {moving_week:03d} -> target {target_week:03d}")
                    continue

                dest_dir = registered_week_dir(output_dir, patient_name, target_week, moving_week)
                dest_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Registering week {moving_week:03d} -> target {target_week:03d}")

                for mod in MODALITIES:
                    moving_path = week_map[moving_week][mod]
                    moving_img = nib.load(moving_path)
                    moving_vol = normalize_volume(moving_img.get_fdata())
                    registered_vol = register_to_reference(moving_vol, fixed_ct1)
                    output_path = dest_dir / f"{MODALITY_FILE_PREFIX[mod]}_skull_strip_registered.nii.gz"
                    nib.save(
                        nib.Nifti1Image(registered_vol.astype(np.float32), fixed_ct1_img.affine, fixed_ct1_img.header),
                        output_path,
                    )

                manifest["jobs"].append(
                    {
                        "patient_id": patient_name,
                        "moving_week": moving_week,
                        "target_week": target_week,
                        "output_dir": str(dest_dir),
                    }
                )

    manifest_path = output_dir / "registration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

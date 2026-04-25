import nibabel as nib
import numpy as np
from pathlib import Path

def generate_dummy_nifti(path: Path, shape=(64, 64, 64)):
    data = np.random.rand(*shape).astype(np.float32)
    # Add a "tumor" center
    center = [s // 2 for s in shape]
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 < 10**2
    data[mask] += 0.5
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)
    print(f"Saved {path}")

def main():
    base_dir = Path("test_data/Patient-999")
    modalities = ["flair", "t1", "t2", "ct1"]
    weeks = ["001", "002"]
    
    for week in weeks:
        skull_strip_path = base_dir / f"week-{week}" / "DeepBraTumIA-segmentation" / "atlas" / "skull_strip"
        skull_strip_path.mkdir(parents=True, exist_ok=True)
        for mod in modalities:
            file_path = skull_strip_path / f"{mod}_skull_strip.nii.gz"
            generate_dummy_nifti(file_path)

if __name__ == "__main__":
    main()

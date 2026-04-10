import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
CONFIG = {
    'root_dir': '/Users/tushar/Documents/Glioblastoma/patient_067',
    'model_path': 'physics_glioma_model_v2.pth',
    'resize_dim': (64, 64, 64),
    'modalities': ('FLAIR', 'T1', 'T2', 'CT1'),
    'start_week': 152
}
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= MODEL CLASSES (Must match training) =================
class ParameterEstimator3D(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU()
        )
        self.to_params = nn.Conv3d(32, 2, 1) 
    def forward(self, x):
        features = self.enc(x)
        raw = self.to_params(features)
        return torch.sigmoid(raw[:, 0:1])*0.5, torch.sigmoid(raw[:, 1:2])*0.5

class TherapySolver(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        k = torch.zeros(1, 1, 3, 3, 3)
        k[0,0,1,1,1] = -6.0
        k[0,0,1,1,0] = 1.0; k[0,0,1,1,2] = 1.0
        k[0,0,1,0,1] = 1.0; k[0,0,1,2,1] = 1.0
        k[0,0,0,1,1] = 1.0; k[0,0,2,1,1] = 1.0
        self.laplacian_k = k.to(device)

    def forward(self, c_init, D, rho, dt, steps=30, therapy_map=None):
        delta_t = dt / steps
        c = c_init
        
        for _ in range(steps):
            laplacian = F.conv3d(c, self.laplacian_k, padding=1)
            
            # Physics: Diffusion + Proliferation
            growth = (D * laplacian) + (rho * c * (1.0 - c))
            
            # Therapy: Subtraction based on radiation map
            kill = 0
            if therapy_map is not None:
                # Kill rate proportional to radiation intensity * tumor concentration
                kill = therapy_map * c * 2.0 # 2.0 is an arbitrary "effectiveness" constant
                
            c = c + (growth - kill) * delta_t
            c = torch.clamp(c, 0.0, 1.0)
            
        return c

# ================= EXECUTION =================
def create_radiation_plan(shape, target_center):
    """Creates a spherical radiation zone at the tumor center"""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = target_center
    radius = 10 # Radiation beam radius
    dist = np.sqrt((x-center[2])**2 + (y-center[1])**2 + (z-center[0])**2)
    
    rad_map = np.zeros(shape)
    rad_map[dist <= radius] = 1.0 # High dose in target
    # Blur edges for realism
    return torch.tensor(rad_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def main():
    # 1. Load Model & Data
    est = ParameterEstimator3D().to(DEVICE)
    solver = TherapySolver(DEVICE).to(DEVICE)
    est.load_state_dict(torch.load(CONFIG['model_path'], map_location=DEVICE), strict=False)
    
    # Load Scan
    channels = []
    for mod in CONFIG['modalities']:
        path = Path(CONFIG['root_dir']) / f"{mod}_wk{CONFIG['start_week']}.nii"
        img = nib.load(path).get_fdata().astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        channels.append(torch.tensor(img))
    x_t0 = torch.stack(channels).unsqueeze(0).to(DEVICE)
    x_t0 = F.interpolate(x_t0, size=CONFIG['resize_dim'], mode='trilinear')
    
    # 2. Estimate Physics
    with torch.no_grad():
        D, rho = est(x_t0)
        c_t0 = x_t0[:, 0:1]
        
        # 3. Create Therapy Plan (Target the center of the image)
        rad_map = create_radiation_plan(CONFIG['resize_dim'], (32, 32, 32)).to(DEVICE)
        
        # 4. Simulate: Natural vs Treated (+6 Months)
        dt_6mo = 24.0 / 52.0
        c_natural = solver(c_t0, D, rho, dt=dt_6mo)
        c_treated = solver(c_t0, D, rho, dt=dt_6mo, therapy_map=rad_map)

    # 5. Visualize
    mid = 32
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].imshow(c_t0[0,0,mid].cpu(), cmap='gray')
    ax[0].set_title("Start (Week 152)")
    
    ax[1].imshow(c_natural[0,0,mid].cpu(), cmap='magma', vmin=0, vmax=1)
    ax[1].set_title("Natural Growth (+6 Mo)")
    
    ax[2].imshow(rad_map[0,0,mid].cpu(), cmap='winter', alpha=0.5)
    ax[2].imshow(c_treated[0,0,mid].cpu(), cmap='magma', alpha=0.7, vmin=0, vmax=1)
    ax[2].set_title("With Radiation (+6 Mo)")
    
    # Show the difference (Benefits of treatment)
    diff = c_natural - c_treated
    ax[3].imshow(diff[0,0,mid].cpu(), cmap='inferno')
    ax[3].set_title("Tumor Reduction Mass")
    
    plt.savefig("therapy_simulation.png")
    plt.show()

if __name__ == "__main__":
    main()
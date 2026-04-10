import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ================= Configuration =================
# MUST match the settings used during training
CONFIG = {
    'root_dir': '/Users/tushar/Documents/Glioblastoma/patient_067',
    'model_path': 'physics_glioma_model_v2.pth',
    'resize_dim': (64, 64, 64),
    'modalities': ('FLAIR', 'T1', 'T2', 'CT1'),
    'last_week': 152  # The starting point for our forecast
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= Model Definitions (Must match training) =================
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
        raw_params = self.to_params(features)
        D = torch.sigmoid(raw_params[:, 0:1]) * 0.5 
        rho = torch.sigmoid(raw_params[:, 1:2]) * 0.5
        return D, rho

class ReactionDiffusionSolver(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        k = torch.zeros(1, 1, 3, 3, 3)
        k[0,0,1,1,1] = -6.0
        k[0,0,1,1,0] = 1.0; k[0,0,1,1,2] = 1.0
        k[0,0,1,0,1] = 1.0; k[0,0,1,2,1] = 1.0
        k[0,0,0,1,1] = 1.0; k[0,0,2,1,1] = 1.0
        self.laplacian_k = k.to(device)

    def forward(self, c_init, D_map, rho_map, dt, steps=20):
        delta_t = dt / steps
        c = c_init
        for _ in range(steps):
            laplacian_c = F.conv3d(c, self.laplacian_k, padding=1)
            diffusion = D_map * laplacian_c
            reaction = rho_map * c * (1.0 - c)
            c = c + (diffusion + reaction) * delta_t
            c = torch.clamp(c, 0.0, 1.0)
        return c

class GliomaPhysicsModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.estimator = ParameterEstimator3D(in_channels=4)
        self.solver = ReactionDiffusionSolver(device)
        
    def forward(self, x_t0, dt):
        D, rho = self.estimator(x_t0)
        c_t0 = x_t0[:, 0:1] 
        # Increase steps for stability during long forecasts
        c_t1_pred = self.solver(c_t0, D, rho, dt=dt, steps=50) 
        return c_t1_pred, D, rho

# ================= Prediction Logic =================
def load_scan(root_dir, week):
    channels = []
    for mod in CONFIG['modalities']:
        path = Path(root_dir) / f"{mod}_wk{week}.nii"
        if not path.exists():
            print(f"❌ File not found: {path}")
            return None
        img = nib.load(path).get_fdata().astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        channels.append(torch.tensor(img))
    
    vol = torch.stack(channels).unsqueeze(0) # (1, C, H, W, D)
    vol = F.interpolate(vol, size=CONFIG['resize_dim'], mode='trilinear', align_corners=False)
    return vol.to(DEVICE)

def main():
    print(f"🔮 Loading Physics Model from {CONFIG['model_path']}...")
    model = GliomaPhysicsModel(DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=DEVICE))
    model.eval()

    print(f"📥 Loading scan for Week {CONFIG['last_week']}...")
    x_t0 = load_scan(CONFIG['root_dir'], CONFIG['last_week'])
    if x_t0 is None: return

    # --- FORECAST PARAMETERS ---
    # We will simulate:
    # 1. Short Term (+3 months / 12 weeks)
    # 2. Long Term (+6 months / 24 weeks)
    dt_short = 12.0 / 52.0  # 12 weeks in years
    dt_long = 24.0 / 52.0   # 24 weeks in years

    print("🚀 Running Physics Simulation...")
    with torch.no_grad():
        # Get biological parameters from the scan
        D_map, rho_map = model.estimator(x_t0)
        
        # Current Tumor
        c_current = x_t0[:, 0:1]
        
        # Simulate Short Term
        c_short = model.solver(c_current, D_map, rho_map, dt=dt_short, steps=30)
        
        # Simulate Long Term (Start from short term to continue growth)
        # Note: We simulate the REMAINING time (Long - Short)
        dt_remaining = dt_long - dt_short
        c_long = model.solver(c_short, D_map, rho_map, dt=dt_remaining, steps=30)

    # --- VISUALIZATION ---
    mid = CONFIG['resize_dim'][0] // 2
    
    # Convert to numpy
    img_current = c_current[0, 0, mid].cpu().numpy()
    img_short = c_short[0, 0, mid].cpu().numpy()
    img_long = c_long[0, 0, mid].cpu().numpy()
    img_D = D_map[0, 0, mid].cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    
    ax[0].imshow(img_current, cmap='gray')
    ax[0].set_title(f"Current Status\n(Week {CONFIG['last_week']})")
    
    ax[1].imshow(img_short, cmap='magma', vmin=0, vmax=1)
    ax[1].set_title(f"Forecast: +3 Months\n(Week {CONFIG['last_week'] + 12})")
    
    ax[2].imshow(img_long, cmap='magma', vmin=0, vmax=1)
    ax[2].set_title(f"Forecast: +6 Months\n(Week {CONFIG['last_week'] + 24})")
    
    im3 = ax[3].imshow(img_D, cmap='jet')
    ax[3].set_title("Underlying Physics\n(Diffusion Map)")
    plt.colorbar(im3, ax=ax[3])
    
    plt.suptitle(f"Patient 067: Glioblastoma Growth Forecast", fontsize=16)
    plt.tight_layout()
    plt.savefig("tumor_forecast.png")
    plt.show()
    print("✅ Forecast generated: tumor_forecast.png")

if __name__ == "__main__":
    main()
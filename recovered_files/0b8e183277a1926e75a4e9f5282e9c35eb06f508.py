import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Device Setup
# ==========================================
CONFIG = {
    'root_dir': '/Users/tushar/Documents/Glioblastoma/patient_067',
    'resize_dim': (64, 64, 64),  # Downsample to 64^3 to fit in VRAM/MPS
    'modalities': ('FLAIR', 'T1', 'T2', 'CT1'),
    'batch_size': 1, 
    'lr': 1e-3,
    'epochs': 150 # Increased epochs slightly to allow physics to settle
}

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"✅ Using device: {DEVICE}")

# ==========================================
# 2. 3D Dataset Loader (Physics-Ready)
# ==========================================
class Glioma3DDataset(Dataset):
    def __init__(self, root_dir, modalities, resize_dim=(64, 64, 64)):
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.resize_dim = resize_dim
        self.valid_weeks = []
        
        if self.root_dir.exists():
            all_weeks = []
            for mod_file in self.root_dir.glob("FLAIR_wk*.nii"):
                try:
                    week = self._extract_week(mod_file.name)
                    all_weeks.append(week)
                except: continue
            
            for week in sorted(all_weeks):
                if all((self.root_dir / f"{m}_wk{week}.nii").exists() for m in modalities):
                    self.valid_weeks.append(week)
        
        self.pairs = []
        for i in range(len(self.valid_weeks) - 1):
            t0 = self.valid_weeks[i]
            t1 = self.valid_weeks[i+1]
            dt = (t1 - t0) / 52.0  # Time in years
            self.pairs.append((t0, t1, dt))

        print(f"Found {len(self.valid_weeks)} scans. Created {len(self.pairs)} training pairs.")

    def _extract_week(self, filename):
        return int(re.search(r'wk(\d+)', filename).group(1))

    def _load_volume(self, week):
        channels = []
        for mod in self.modalities:
            path = self.root_dir / f"{mod}_wk{week}.nii"
            img = nib.load(path).get_fdata().astype(np.float32)
            # Normalize to 0-1
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            channels.append(torch.tensor(img))
            
        volume = torch.stack(channels).unsqueeze(0) # (1, C, H, W, D)
        volume = F.interpolate(volume, size=self.resize_dim, mode='trilinear', align_corners=False)
        return volume.squeeze(0) 

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        week_t0, week_t1, dt = self.pairs[idx]
        return {
            'x_t0': self._load_volume(week_t0),
            'y_t1': self._load_volume(week_t1)[0:1], # Target is just FLAIR
            'dt': torch.tensor(dt, dtype=torch.float32),
            'weeks': (week_t0, week_t1)
        }

# ==========================================
# 3. The Learnable Physics Engine (UPDATED)
# ==========================================

class ParameterEstimator3D(nn.Module):
    """
    Inputs: 4-Channel MRI
    Outputs: 2 Biological Parameter Maps (Diffusion D, Proliferation Rho)
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        self.to_params = nn.Conv3d(32, 2, 1) 

    def forward(self, x):
        features = self.enc(x)
        raw_params = self.to_params(features)
        
        # FIX 1: Increased Cap for Diffusion
        # Previously * 0.1, now * 0.5 to prevent "flat red map" saturation
        D = torch.sigmoid(raw_params[:, 0:1]) * 0.5 
        
        # Proliferation (rho)
        rho = torch.sigmoid(raw_params[:, 1:2]) * 0.5
        
        return D, rho

class ReactionDiffusionSolver(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # 3D Laplacian Kernel
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
            # Diffusion: D * Laplacian(c)
            laplacian_c = F.conv3d(c, self.laplacian_k, padding=1)
            diffusion = D_map * laplacian_c
            
            # Reaction: rho * c * (1-c)
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
        c_t0 = x_t0[:, 0:1] # FLAIR channel as tumor concentration
        
        # Pass mean dt (assuming batch items have similar dt, or batch_size=1)
        c_t1_pred = self.solver(c_t0, D, rho, dt=dt.mean(), steps=20)
        
        return c_t1_pred, D, rho

# ==========================================
# 4. Training & Visualization (UPDATED)
# ==========================================

def train_physics_model():
    dataset = Glioma3DDataset(CONFIG['root_dir'], CONFIG['modalities'], CONFIG['resize_dim'])
    
    if len(dataset) == 0:
        print("❌ No valid data pairs found.")
        return None, []

    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    model = GliomaPhysicsModel(DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    mse_crit = nn.MSELoss()
    
    print("\n🚀 Starting Physics-Informed Training (v2)...")
    loss_history = []
    
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        
        for batch in loader:
            x_t0 = batch['x_t0'].to(DEVICE)
            y_t1 = batch['y_t1'].to(DEVICE)
            dt = batch['dt'].to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_t1, D_map, rho_map = model(x_t0, dt)
            
            # --- FIX 2: Updated Loss Function ---
            fit_loss = mse_crit(pred_t1, y_t1)
            
            # Smoothness (Regularization)
            smooth_loss = torch.mean(torch.abs(D_map[:, :, 1:] - D_map[:, :, :-1])) + \
                          torch.mean(torch.abs(rho_map[:, :, 1:] - rho_map[:, :, :-1]))
            
            # Contrast Reward: Negative Variance
            # Punishes the model if D_map is a flat color. Forces it to find structure.
            contrast_reward = -1.0 * torch.var(D_map)
            
            # Weights: 1.0 Fit | 0.1 Smoothness | 0.05 Contrast Reward
            total_loss = fit_loss + (0.1 * smooth_loss) + (0.05 * contrast_reward)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            # Monitor D_map min/max to ensure it's not sticking to caps
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | D_rng: [{D_map.min():.3f}-{D_map.max():.3f}]")

    return model, loss_history

def visualize_results(model, dataset):
    idx = len(dataset) - 1
    data = dataset[idx]
    
    x_t0 = data['x_t0'].unsqueeze(0).to(DEVICE)
    y_t1 = data['y_t1'].unsqueeze(0).to(DEVICE)
    dt = data['dt'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_t1, D_map, rho_map = model(x_t0, dt)
    
    # Slice the middle of the 3D volume
    mid = CONFIG['resize_dim'][0] // 2
    
    t0_img = x_t0[0, 0, mid].cpu().numpy()
    pred_img = pred_t1[0, 0, mid].cpu().numpy()
    true_img = y_t1[0, 0, mid].cpu().numpy()
    diff_map = D_map[0, 0, mid].cpu().numpy()
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].imshow(t0_img, cmap='gray')
    ax[0].set_title(f"Input Week {data['weeks'][0]}")
    
    ax[1].imshow(pred_img, cmap='magma')
    ax[1].set_title(f"Simulation Week {data['weeks'][1]}")
    
    ax[2].imshow(true_img, cmap='gray')
    ax[2].set_title(f"Ground Truth Week {data['weeks'][1]}")
    
    # Visualizing the inferred biology
    im3 = ax[3].imshow(diff_map, cmap='jet')
    ax[3].set_title("Inferred Diffusion (Invasion Map)")
    plt.colorbar(im3, ax=ax[3])
    
    plt.show()

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    if not Path(CONFIG['root_dir']).exists():
        print(f"⚠️ PATH ERROR: {CONFIG['root_dir']} not found.")
    else:
        trained_model, history = train_physics_model()
        
        if trained_model:
            torch.save(trained_model.state_dict(), "physics_glioma_model_v2.pth")
            
            # Visualize
            dataset = Glioma3DDataset(CONFIG['root_dir'], CONFIG['modalities'], CONFIG['resize_dim'])
            visualize_results(trained_model, dataset)
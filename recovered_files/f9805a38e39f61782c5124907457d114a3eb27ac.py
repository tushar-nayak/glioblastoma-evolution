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
    'batch_size': 1,  # 3D is heavy, keep batch size small
    'lr': 1e-3,
    'epochs': 100
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
        
        # --- Logic from previous code to find valid weeks ---
        if self.root_dir.exists():
            all_weeks = []
            for mod_file in self.root_dir.glob("FLAIR_wk*.nii"):
                try:
                    week = self._extract_week(mod_file.name)
                    all_weeks.append(week)
                except: continue
            
            # Filter only weeks that have ALL modalities
            for week in sorted(all_weeks):
                if all((self.root_dir / f"{m}_wk{week}.nii").exists() for m in modalities):
                    self.valid_weeks.append(week)
        
        # Create pairs: (Week T, Week T+N)
        self.pairs = []
        for i in range(len(self.valid_weeks) - 1):
            # We predict the immediate next available scan
            # In a real physics model, we need the delta_t (time difference)
            t0 = self.valid_weeks[i]
            t1 = self.valid_weeks[i+1]
            dt = (t1 - t0) / 52.0  # Time in years (approx)
            self.pairs.append((t0, t1, dt))

        print(f"Found {len(self.valid_weeks)} scans. Created {len(self.pairs)} training pairs.")

    def _extract_week(self, filename):
        return int(re.search(r'wk(\d+)', filename).group(1))

    def _load_volume(self, week):
        channels = []
        for mod in self.modalities:
            path = self.root_dir / f"{mod}_wk{week}.nii"
            # Load data: (H, W, D)
            img = nib.load(path).get_fdata().astype(np.float32)
            
            # Normalize (0 to 1 range is crucial for Physics stability)
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            
            channels.append(torch.tensor(img))
            
        # Stack: (C, H, W, D)
        volume = torch.stack(channels)
        
        # Add batch dim for interpolation: (1, C, H, W, D)
        volume = volume.unsqueeze(0)
        
        # Resize to 64x64x64
        volume = F.interpolate(volume, size=self.resize_dim, mode='trilinear', align_corners=False)
        
        return volume.squeeze(0) # Remove batch dim -> (C, 64, 64, 64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        week_t0, week_t1, dt = self.pairs[idx]
        
        vol_t0 = self._load_volume(week_t0)
        vol_t1 = self._load_volume(week_t1)
        
        # We assume Channel 0 (FLAIR) is the "Tumor Concentration" proxy
        # The other channels help predict parameters D and Rho
        return {
            'x_t0': vol_t0,       # Full MRI at T0
            'y_t1': vol_t1[0:1],  # Target: Just FLAIR at T1 (Tumor mass)
            'dt': torch.tensor(dt, dtype=torch.float32),
            'weeks': (week_t0, week_t1)
        }

# ==========================================
# 3. The Learnable Physics Engine
# ==========================================

class ParameterEstimator3D(nn.Module):
    """
    Inputs: 4-Channel MRI (FLAIR, T1, T2, CT1)
    Outputs: 2 Biological Parameter Maps (Diffusion D, Proliferation Rho)
    """
    def __init__(self, in_channels=4):
        super().__init__()
        # Simple 3D Encoder-Decoder to map Anatomy -> Biology
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        self.to_params = nn.Conv3d(32, 2, 1) # Output 2 channels

    def forward(self, x):
        features = self.enc(x)
        raw_params = self.to_params(features)
        
        # Enforce physical constraints: D and Rho must be positive
        # D (Diffusion): usually small. Scale by 0.1
        # Rho (Proliferation): Scale by 0.5
        D = torch.sigmoid(raw_params[:, 0:1]) * 0.1
        rho = torch.sigmoid(raw_params[:, 1:2]) * 0.5
        
        return D, rho

class ReactionDiffusionSolver(nn.Module):
    """
    Differentiable PDE Solver for the Fisher-Kolmogorov Equation:
    dc/dt = Div(D * Grad(c)) + rho * c * (1 - c)
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        # 3D Laplacian Kernel (Finite Difference)
        k = torch.zeros(1, 1, 3, 3, 3)
        k[0,0,1,1,1] = -6.0
        k[0,0,1,1,0] = 1.0; k[0,0,1,1,2] = 1.0
        k[0,0,1,0,1] = 1.0; k[0,0,1,2,1] = 1.0
        k[0,0,0,1,1] = 1.0; k[0,0,2,1,1] = 1.0
        self.laplacian_k = k.to(device)

    def forward(self, c_init, D_map, rho_map, dt, steps=10):
        # We subdivide the time delta 'dt' into smaller stability steps
        # Courant-Friedrichs-Lewy (CFL) condition requires small steps
        delta_t = dt / steps
        c = c_init
        
        for _ in range(steps):
            # 1. Diffusion Term: D * Laplacian(c)
            # Note: A full implementation computes Div(D * Grad(c)), 
            # but D * Laplacian(c) is a standard approximation for smooth D.
            laplacian_c = F.conv3d(c, self.laplacian_k, padding=1)
            diffusion = D_map * laplacian_c
            
            # 2. Reaction Term: rho * c * (1 - c) (Logistic Growth)
            reaction = rho_map * c * (1.0 - c)
            
            # 3. Euler Integration
            dc = diffusion + reaction
            c = c + dc * delta_t
            
            # 4. Clamp to physically valid range [0, 1]
            c = torch.clamp(c, 0.0, 1.0)
            
        return c

class GliomaPhysicsModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.estimator = ParameterEstimator3D(in_channels=4)
        self.solver = ReactionDiffusionSolver(device)
        
    def forward(self, x_t0, dt):
        # 1. Estimate Biology from Anatomy
        D, rho = self.estimator(x_t0)
        
        # 2. Extract initial tumor concentration (FLAIR is channel 0)
        c_t0 = x_t0[:, 0:1] 
        
        # 3. Simulate Growth
        # We pass dt as a scalar, expanding it inside solver is handled by logic
        # Here we assume dt is a tensor of shape (B,)
        # For batch=1, we just take the item.
        c_t1_pred = self.solver(c_t0, D, rho, dt=dt.mean(), steps=20)
        
        return c_t1_pred, D, rho

# ==========================================
# 4. Training & Visualization
# ==========================================

def train_physics_model():
    # Setup
    dataset = Glioma3DDataset(CONFIG['root_dir'], CONFIG['modalities'], CONFIG['resize_dim'])
    
    if len(dataset) == 0:
        print("❌ No valid data pairs found. Check path.")
        return

    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    model = GliomaPhysicsModel(DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # Loss: MSE + Physics Regularization
    mse_crit = nn.MSELoss()
    
    print("\n🚀 Starting Physics-Informed Training...")
    
    loss_history = []
    
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        
        for batch in loader:
            x_t0 = batch['x_t0'].to(DEVICE)
            y_t1 = batch['y_t1'].to(DEVICE) # True future tumor
            dt = batch['dt'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward: Learn params -> Simulate physics
            pred_t1, D_map, rho_map = model(x_t0, dt)
            
            # Loss Calculation
            # 1. Prediction Error
            fit_loss = mse_crit(pred_t1, y_t1)
            
            # 2. Regularization (Keep D and rho smooth, biology doesn't change abruptly)
            smooth_loss = torch.mean(torch.abs(D_map[:, :, 1:] - D_map[:, :, :-1])) + \
                          torch.mean(torch.abs(rho_map[:, :, 1:] - rho_map[:, :, :-1]))
            
            total_loss = fit_loss + 0.1 * smooth_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | Min D: {D_map.min():.4f} | Max Rho: {rho_map.max():.4f}")

    return model, loss_history

def visualize_results(model, dataset):
    """
    Takes the last sample in the dataset and visualizes:
    1. Input Tumor (T0)
    2. Predicted Tumor (T1)
    3. Actual Tumor (T1)
    4. Inferred Diffusion Map (The 'Why')
    """
    idx = len(dataset) - 1
    data = dataset[idx]
    
    x_t0 = data['x_t0'].unsqueeze(0).to(DEVICE)
    y_t1 = data['y_t1'].unsqueeze(0).to(DEVICE)
    dt = data['dt'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_t1, D_map, rho_map = model(x_t0, dt)
    
    # Slice the middle of the 3D volume
    mid = CONFIG['resize_dim'][0] // 2
    
    # Convert to numpy
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
    
    # Visualize the Biology the model learned
    im3 = ax[3].imshow(diff_map, cmap='jet')
    ax[3].set_title("Inferred Diffusion (Invasion Map)")
    plt.colorbar(im3, ax=ax[3])
    
    plt.show()

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    # Check if path exists
    if not Path(CONFIG['root_dir']).exists():
        print(f"⚠️ PATH ERROR: {CONFIG['root_dir']} not found.")
    else:
        trained_model, history = train_physics_model()
        
        # Save
        torch.save(trained_model.state_dict(), "physics_glioma_model.pth")
        
        # Visualize
        dataset = Glioma3DDataset(CONFIG['root_dir'], CONFIG['modalities'], CONFIG['resize_dim'])
        visualize_results(trained_model, dataset)
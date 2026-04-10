import re
from pathlib import Path
import torch
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

import torch

def get_device():
    """Checks for MPS, then falls back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# Check for MPS availability
mps_is_available = torch.backends.mps.is_available()
mps_is_built = torch.backends.mps.is_built()

print(f"MPS available: {mps_is_available}")
print(f"MPS built (part of PyTorch install): {mps_is_built}")

if mps_is_available and mps_is_built:
    # Set the device to 'mps'
    device = torch.device("mps")
    print("\n✅ Using Apple Silicon MPS device for acceleration.")
else:
    # Fallback to CPU if MPS is not ready
    device = torch.device("cpu")
    print("\n⚠️ MPS not available. Falling back to CPU.")

# Example to show it's working: create a tensor on the MPS device
if str(device) == 'mps':
    x = torch.rand(5, 5).to(device)
    print(f"\nExample tensor created on: {x.device}")

class LUMIEREDataset(Dataset):
    def __init__(self, root_dir, modalities=('FLAIR', 'T1', 'T2', 'CT1'), 
                 norm_params=None, target_patient='patient_067', 
                 week_range=(90, 160), slice_range=(70, 90)):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.target_patient = target_patient
        self.week_range = week_range
        self.slice_range = slice_range
        self.norm_params = norm_params or {
            'CT1': {'mean': 0.94, 'std': 0.17},
            'T1': {'mean': 0.71, 'std': 0.21},
            'T2': {'mean': 0.70, 'std': 0.26},
            'FLAIR': {'mean': 0.60, 'std': 0.21}
        }

        self.valid_weeks = []
        if self.root_dir.exists() and self.root_dir.is_dir():
            all_weeks = []
            for mod_file in self.root_dir.glob(f"FLAIR_wk*.nii"):
                try:
                    week_num = self._extract_week_number(mod_file.name)
                    if self.week_range[0] <= week_num <= self.week_range[1]:
                        all_weeks.append(week_num)
                except ValueError:
                    continue
            
            for week in sorted(all_weeks):
                all_mods_exist = True
                for mod in self.modalities:
                    if not (self.root_dir / f"{mod}_wk{week}.nii").exists():
                        all_mods_exist = False
                        break
                
                if all_mods_exist:
                    self.valid_weeks.append(week)
        
        print(f"Found {len(self.valid_weeks)} valid weeks for {self.target_patient}")
        if self.valid_weeks:
            print(f"Weeks found: {self.valid_weeks}")

    def __len__(self):
        return len(self.valid_weeks)

    def __getitem__(self, idx):
        week_num = self.valid_weeks[idx]
        
        modality_tensors = []
        for mod in self.modalities:
            img_path = self.root_dir / f"{mod}_wk{week_num}.nii"
            data = nib.load(img_path).get_fdata()
            
            # Ensure dimensions are divisible by 16
            h, w = data.shape[:2]
            h = (h // 16) * 16
            w = (w // 16) * 16
            data = data[:h, :w, :]
            
            start_slice = max(0, self.slice_range[0])
            end_slice = min(data.shape[2], self.slice_range[1])
            sliced_data = data[:, :, start_slice:end_slice]
            
            normalized = (sliced_data - self.norm_params[mod]['mean']) / \
                         self.norm_params[mod]['std']
            modality_tensors.append(torch.FloatTensor(normalized))
        
        mri_tensor = torch.stack(modality_tensors)
        mri_tensor = mri_tensor.permute(0, 3, 1, 2)

        return {
            'mri': mri_tensor,
            'week': week_num,
            'patient_id': self.target_patient,
            'time_delta': 1.0  
        }

    def _extract_week_number(self, filename):
        match = re.search(r'wk(\d+)', filename)
        return int(match.group(1)) if match else ValueError(f"Invalid filename: {filename}")

class TimeAwareDatasetWrapper:
    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.week_pairs = self._create_multi_week_pairs()
        
    def _create_multi_week_pairs(self):
        valid_weeks = sorted(self.dataset.valid_weeks)
        pairs = []
        
        for i in range(len(valid_weeks)-1):
            context_weeks = valid_weeks[i:min(i+3, len(valid_weeks))]
            if len(context_weeks) < 2:
                continue
                
            target_indices = [j for j in range(len(valid_weeks)) 
                             if valid_weeks[j] > context_weeks[-1]]
            
            for target_idx in target_indices:
                target = valid_weeks[target_idx]
                time_delta = (target - context_weeks[-1]) / 52.0 
                pairs.append({
                    'context_weeks': context_weeks,
                    'target_week': target,
                    'time_delta': time_delta
                })
        return pairs
    
    def __len__(self):
        return len(self.week_pairs)
    
    def __getitem__(self, idx):
        pair = self.week_pairs[idx]
        
        context_data = []
        for week in pair['context_weeks']:
            data = self.dataset[self.dataset.valid_weeks.index(week)]
            
            middle_slice_idx = data['mri'].shape[1] // 2
            slice_start = max(0, middle_slice_idx-1)
            slice_end = min(data['mri'].shape[1], middle_slice_idx+2)
            selected_slices = data['mri'][:, slice_start:slice_end]
            
            flat_slices = selected_slices.reshape(-1, *selected_slices.shape[2:])
            context_data.append(flat_slices)
            
        context_tensor = torch.cat(context_data, dim=0)
        
        target_data = self.dataset[self.dataset.valid_weeks.index(pair['target_week'])]
        middle_slice_idx = target_data['mri'].shape[1] // 2
        target_slice = target_data['mri'][:, middle_slice_idx]
        
        return {
            'context': context_tensor,
            'target': target_slice,
            'time_delta': torch.tensor(pair['time_delta'], dtype=torch.float32),
            'context_weeks': pair['context_weeks']
        }




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled

class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, g, s):
        if g.shape[-2:] != s.shape[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode='bilinear', align_corners=True)
            
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        attention = self.output(out)
        return attention * s

class DecoderBlock(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention = AttentionGate(g_channels, s_channels, out_channels)
        self.conv = ConvBlock(g_channels + s_channels, out_channels)
    
    def forward(self, g, s):
        g = self.up(g)
        
        if g.size()[-2:] != s.size()[-2:]:
            g = F.interpolate(g, size=s.shape[-2:], mode='bilinear', align_corners=True)
            
        s = self.attention(g, s)
        combined = torch.cat([g, s], dim=1)
        return self.conv(combined)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.encoder1 = EncoderBlock(in_channels, features[0])
        self.encoder2 = EncoderBlock(features[0], features[1])
        self.encoder3 = EncoderBlock(features[1], features[2])
        self.encoder4 = EncoderBlock(features[2], features[3])
        
        self.bottleneck = ConvBlock(features[3], features[3]*2)
        
        self.decoder1 = DecoderBlock(features[3]*2, features[3], features[3])
        self.decoder2 = DecoderBlock(features[3], features[2], features[2])
        self.decoder3 = DecoderBlock(features[2], features[1], features[1])
        self.decoder4 = DecoderBlock(features[1], features[0], features[0])
        
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
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
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )

    def forward(self, t, y):
        t_vector = self.time_embed(t.view(1,1))
        t_expanded = t_vector.view(1, -1, 1, 1).expand(y.size(0), -1, y.size(2), y.size(3))
        return self.conv_block(torch.cat([y, t_expanded], dim=1))

class TemporalEncoder(nn.Module):
    def __init__(self, max_weeks=160):
        super().__init__()
        self.week_embed = nn.Embedding(max_weeks, 64)
        
    def forward(self, week_numbers):
        if not isinstance(week_numbers, torch.Tensor):
            week_numbers = torch.tensor(week_numbers, dtype=torch.long)
        return self.week_embed(week_numbers.long())

class GliomaGrowthPredictor(nn.Module):
    def __init__(self, input_channels=12, output_channels=4, hidden_dim=64, time_aware=True):
        super().__init__()
        self.time_aware = time_aware
        
        self.unet = AttentionUNet(input_channels, hidden_dim)
        
        if self.time_aware:
            self.temporal_encoder = TemporalEncoder()
        
        self.ode_func = TimeAwareODEFunc(hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, output_channels, 1)  
        )
        
        self.min_week = 90
        self.max_week = 160 

    def forward(self, x, t_delta, context_weeks=None):
        features = self.unet(x)
        
        if self.time_aware and context_weeks is not None:
            # Move context_weeks tensor to device before embedding lookup
            week_tensor = torch.tensor(context_weeks, dtype=torch.long, device=x.device)
            temporal_emb = self.temporal_encoder(week_tensor).mean(dim=0)
            features += temporal_emb.view(1, -1, 1, 1)
        
        t_normalized = t_delta / (self.max_week - self.min_week)
        
        # 1. Ensure t_span is explicitly float32
        t_span = torch.linspace(0, t_normalized.item(), steps=5, dtype=torch.float32, device=x.device)
        
        # 2. CRITICAL FIX: Switch solver to 'midpoint' (or 'euler') to avoid internal float64 issues 
        # with 'dopri5' on the MPS backend. Remove rtol/atol as they are not needed for fixed-step solvers.
        solution = odeint(
            self.ode_func, 
            features, 
            t_span, 
            method='midpoint', # <--- CHANGED SOLVER HERE
            # Note: rtol and atol are not used for 'midpoint' and can be omitted.
            # However, ensure that features (y0) and t_span are float32, which they are.
        )
        
        return self.decoder(solution[-1])    

epx = 7500
def train_model(model, dataset_wrapper, epochs=epx, lr=1e-6, device='mps'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dice_loss = lambda pred, target: 1 - (2*(pred*target).sum())/(pred.sum()+target.sum()+1e-8)
    mse_loss = nn.MSELoss()
    
    dataloader = DataLoader(dataset_wrapper, batch_size=1, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            time_delta = batch['time_delta'].to(device)
            context_weeks = batch['context_weeks']
            
            optimizer.zero_grad()
            prediction = model(context, time_delta, context_weeks)
            
            loss = mse_loss(prediction, target) + dice_loss(torch.sigmoid(prediction), target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    return model

def predict_tumor_growth(model, dataset, from_weeks, to_week, device='cuda'):
    model.eval()
    
    context_data = []
    for week in from_weeks:
        week_idx = dataset.valid_weeks.index(week)
        data = dataset[week_idx]
        
        middle_slice_idx = data['mri'].shape[1] // 2
        slice_start = max(0, middle_slice_idx-1)
        slice_end = min(data['mri'].shape[1], middle_slice_idx+2)
        selected_slices = data['mri'][:, slice_start:slice_end]

        flat_slices = selected_slices.reshape(-1, *selected_slices.shape[2:])
        context_data.append(flat_slices)
    
    context_tensor = torch.cat(context_data, dim=0).unsqueeze(0).to(device)
    time_delta = torch.tensor([(to_week - from_weeks[-1])/52.0], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction = model(context_tensor, time_delta, from_weeks)
    
    return prediction

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = LUMIEREDataset(
        root_dir='/Users/tushar/Documents/Glioblastoma/patient_067',
        modalities=('FLAIR', 'T1', 'T2', 'CT1'),
        week_range=(90, 160),
        slice_range=(70, 90)  
    )
    wrapped_dataset = TimeAwareDatasetWrapper(dataset)
    
    print(f"Created dataset wrapper with {len(wrapped_dataset)} prediction pairs")
    
    sample = wrapped_dataset[0]
    input_channels = sample['context'].shape[0]
    
    model = GliomaGrowthPredictor(
        input_channels=input_channels,
        hidden_dim=64,
        time_aware=True
    )
    
    print("Model created successfully")
    print(f"Context shape: {sample['context'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    
    print("Starting model training...")
    trained_model = train_model(model, wrapped_dataset, epochs=epx, device=device)
    
    torch.save(trained_model.state_dict(), "attention_unet_glioma_model.pth")
    print("Model training completed and saved")
    
    print("Generating prediction example...")
    prediction = predict_tumor_growth(
        model=trained_model,
        dataset=dataset,
        from_weeks=[109, 124, 136],
        to_week=152,
        device=device
    )
    
    print(f"Prediction shape: {prediction.shape}")
    print("Prediction completed")

if __name__ == "__main__":
    main()


dataset = LUMIEREDataset(
    root_dir='/Users/tushar/Documents/Glioblastoma/patient_067',
    modalities=('FLAIR', 'T1', 'T2', 'CT1'),
    week_range=(90, 160),
    slice_range=(70, 90)
)


model = GliomaGrowthPredictor(
    input_channels=36,  
    hidden_dim=64,
    time_aware=True
).to('mps')

# 2. Load trained weights
model.load_state_dict(torch.load("attention_unet_glioma_model.pth"))

# 3. Now use this model for predictions
prediction = predict_tumor_growth(
    model=model,
    dataset=dataset,
    from_weeks=[109, 124, 136],
    to_week=152,
    device='mps'
)


import matplotlib.pyplot as plt
import torch

# Assuming 'prediction' is the model output and 'target' is the ground truth from your dataset
# Get the same target slice for visual comparison
target_idx = dataset.valid_weeks.index(152)
target_data = dataset[target_idx]
target_slice = target_data['mri'][:, target_data['mri'].shape[1] // 2]  # shape: [channels, H, W]

# Move tensors to CPU and convert to numpy for plotting
pred_img = prediction.squeeze().detach().cpu().numpy()
gt_img = target_slice.squeeze().detach().cpu().numpy()

# If you want to display a single modality/channel (e.g., the first one)
channel = 0

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(gt_img[channel], cmap='gray')
plt.title('Ground Truth (Week 152)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pred_img[channel], cmap='gray')
plt.title('Model Prediction')
plt.axis('off')

plt.show()

# ---------------------------- Simplified Evaluation Module ----------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt

class GliomaModelEvaluator:
    def __init__(self, model, dataset, device='mps'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.threshold = 0.5
    
    def calculate_metrics(self, prediction, target):
        """Calculate MSE, volume difference, and Dice metrics"""
        pred_sigmoid = torch.sigmoid(prediction)
        pred_np = pred_sigmoid.detach().cpu().numpy()[0, 0]
        target_np = target.detach().cpu().numpy()[0, 0]
        
        # Mean Squared Error
        mse = ((pred_np - target_np) ** 2).mean()
        
        # Volumetric analysis
        pred_binary = (pred_np > self.threshold).astype(float)
        target_binary = (target_np > self.threshold).astype(float)
        pred_volume = np.sum(pred_binary)
        target_volume = np.sum(target_binary)
        vol_diff = abs(pred_volume - target_volume) / max(target_volume, 1)
        
        # Dice coefficient
        intersection = np.sum(pred_binary * target_binary)
        dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)
        
        return {
            'mse': mse,
            'volume_diff': vol_diff,
            'dice': dice
        }
    
    def evaluate_prediction(self, from_weeks, to_week):
        """
        ✅ THIS METHOD WAS MISSING!
        Evaluate model on specific prediction task
        """
        # Get prediction
        prediction = predict_tumor_growth(
            self.model, self.dataset, from_weeks, to_week, self.device
        )
        
        # Get ground truth
        target_idx = self.dataset.valid_weeks.index(to_week)
        target_data = self.dataset[target_idx]
        target_slice = target_data['mri'][:, target_data['mri'].shape[1] // 2].unsqueeze(0).to(self.device)
        
        # Calculate metrics
        metrics = self.calculate_metrics(prediction, target_slice)
        
        # Visualize
        self.visualize_prediction(prediction, target_slice, from_weeks, to_week)
        
        return metrics
    
    def visualize_prediction(self, prediction, target, from_weeks, to_week):
        """Visualize prediction vs ground truth for all modalities"""
        pred_img = torch.sigmoid(prediction).squeeze().detach().cpu().numpy()
        gt_img = target.squeeze().detach().cpu().numpy()
        
        # Determine number of channels to display (max 4 for FLAIR, T1, T2, CT1)
        n_channels = min(4, pred_img.shape[0])
        
        fig, axes = plt.subplots(2, n_channels, figsize=(16, 8))
        
        modality_names = ['FLAIR', 'T1', 'T2', 'CT1']
        
        for i in range(n_channels):
            # Ground truth
            axes[0, i].imshow(gt_img[i], cmap='gray', vmin=gt_img[i].min(), vmax=gt_img[i].max())
            axes[0, i].set_title(f'Ground Truth - {modality_names[i]}')
            axes[0, i].axis('off')
            
            # Prediction
            axes[1, i].imshow(pred_img[i], cmap='gray', vmin=pred_img[i].min(), vmax=pred_img[i].max())
            axes[1, i].set_title(f'Prediction - {modality_names[i]}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Tumor Growth Prediction\nContext: weeks {from_weeks} → Target: week {to_week}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = f'prediction_from_{from_weeks[0]}_to_{to_week}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_path}")
        plt.show()

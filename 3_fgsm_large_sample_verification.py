"""
=============================================================================
Program Name: 3_fgsm_large_sample_verification.py
Version: 1.1
Description: 
    1. Loads a large sample (10,000 clips) of Speech Commands.
    2. Processes attacks in batches to prevent Colab GPU OOM errors.
    3. Computes the highly stable SNR ratio for FGSM vs. PGD.
    4. Plots the results large_sample_verification.png, zooming in on the k=[10, 20] band to check for divergence.
    5. Loads trained model checkpoint from Google Drive project folder.
GPU Needed: Yes
Dependencies: torch, torchaudio, torchattacks, matplotlib, numpy
=============================================================================
"""

import os
import shutil
import torch
import torch.nn as nn
import torchaudio
import torchattacks
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Configuration for Google Drive ---
PROJECT_DIR = "/content/drive/MyDrive/paper/cepstral_lifter/"  # Persistent storage for checkpoints

def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def load_from_drive(remote_name, local_path):
    """
    Copy a file from Google Drive project folder to local path.
    remote_name: filename in Drive.
    local_path: destination local path.
    Returns True if file exists and copied, False otherwise.
    """
    ensure_project_dir()
    src_path = os.path.join(PROJECT_DIR, remote_name)
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, local_path)
            print(f"  [DRIVE OK] {src_path}  →  {local_path}")
            return True
        except Exception as e:
            print(f"  [DRIVE FAIL] copy from {src_path}: {e}")
            return False
    else:
        print(f"  [DRIVE MISSING] {src_path} not found")
        return False

# --- 1. Recreate the Baseline Model (Needed for gradients) ---
class SpeechCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Hardcode the exact dimension from the checkpoint (64 * 5 * 12 = 3840)
        self.fc = nn.Linear(3840, n_classes) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.mfcc(x) 
        x = x.unsqueeze(1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
        
model = SpeechCNN().to(DEVICE)

# Load trained weights from Google Drive project folder
model_checkpoint = "02_baseline_speech_cnn.pth"
if load_from_drive(model_checkpoint, model_checkpoint):
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    print("Loaded trained baseline model from Drive.")
else:
    print("Warning: Trained model not found in Drive. Using randomly initialized weights. Geometry will still hold, but magnitude may vary.")

model.eval()

# --- 2. Load Data ---
print("Loading Speech Commands dataset...")
# Mount Drive if needed (for accessing dataset)
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

dataset = torchaudio.datasets.SPEECHCOMMANDS(
    root="./data", url="speech_commands_v0.02", folder_in_archive="SpeechCommands", download=True
)

# Extract exactly 10,000 valid 1-second clips
real_clips = []
labels = []
for i in range(len(dataset)):
    waveform, _, label_idx, _, _ = dataset[i]
    if waveform.shape[1] == 16000:
        real_clips.append(waveform.squeeze(0))
        # Just creating dummy integer labels for the attack objective
        labels.append(len(labels) % 10) 
    if len(real_clips) == 10000:
        break

data_tensor = torch.stack(real_clips)
label_tensor = torch.tensor(labels)
dataset = TensorDataset(data_tensor, label_tensor)
# Batch size of 200 is usually safe for T4 GPUs
dataloader = DataLoader(dataset, batch_size=200, shuffle=False) 

# --- 3. Setup Attacks and Trackers ---
attacks = {
    "FGSM": torchattacks.FGSM(model, eps=0.005),
    # Running PGD on a smaller subset (1000) to save time, as iterative attacks are slow
    "PGD-20": torchattacks.PGD(model, eps=0.005, alpha=0.001, steps=20) 
}

accumulated_power = {
    "clean": np.zeros(40),
    "FGSM_adv": np.zeros(40),
    "PGD-20_adv": np.zeros(40)
}

# --- 4. Processing Loop ---
print("Starting batched attack processing...")
for batch_idx, (audio, target) in enumerate(dataloader):
    audio, target = audio.to(DEVICE), target.to(DEVICE)
    
    # Clean profile accumulation
    clean_mfcc = model.mfcc(audio).detach()
    accumulated_power["clean"] += torch.square(clean_mfcc).mean(dim=(0, 2)).cpu().numpy() * len(audio)
    
    # FGSM accumulation (Run on all 10,000 samples)
    adv_audio_fgsm = attacks["FGSM"](audio, target)
    adv_mfcc_fgsm = model.mfcc(adv_audio_fgsm).detach()
    delta_fgsm = adv_mfcc_fgsm - clean_mfcc
    accumulated_power["FGSM_adv"] += torch.square(delta_fgsm).mean(dim=(0, 2)).cpu().numpy() * len(audio)

    # PGD accumulation (Run on first 5 batches / 1000 samples to save time)
    if batch_idx < 5:
        adv_audio_pgd = attacks["PGD-20"](audio, target)
        adv_mfcc_pgd = model.mfcc(adv_audio_pgd).detach()
        delta_pgd = adv_mfcc_pgd - clean_mfcc
        accumulated_power["PGD-20_adv"] += torch.square(delta_pgd).mean(dim=(0, 2)).cpu().numpy() * len(audio)
        
    if (batch_idx + 1) % 10 == 0:
        print(f"Processed {batch_idx * 200 + 200}/10000 clips...")

# Normalize by counts
clean_profile = accumulated_power["clean"] / 10000
fgsm_profile = accumulated_power["FGSM_adv"] / 10000
pgd_profile = accumulated_power["PGD-20_adv"] / 1000 # Only ran 1000

# --- 5. Plotting ---
k_indices = np.arange(1, 41)
fgsm_ratio = fgsm_profile / (clean_profile + 1e-10)
pgd_ratio = pgd_profile / (clean_profile + 1e-10)

# Normalize the ratios to peak=1.0 to check the curve collapse
fgsm_norm = fgsm_ratio / np.max(fgsm_ratio)
pgd_norm = pgd_ratio / np.max(pgd_ratio)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Full Range
ax1.plot(k_indices, fgsm_norm, label="FGSM (n=10,000)", linewidth=2)
ax1.plot(k_indices, pgd_norm, label="PGD-20 (n=1,000)", linewidth=2, linestyle='--')
ax1.set_title("Normalized SNR Ratio (Full Quefrency Range)")
ax1.set_xlabel("Quefrency Index (k)")
ax1.set_ylabel("Normalized Ratio")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Zoomed in on Formant Transition Region k=[10, 20]
zoom_mask = (k_indices >= 10) & (k_indices <= 20)
ax2.plot(k_indices[zoom_mask], fgsm_norm[zoom_mask], label="FGSM (n=10,000)", linewidth=2, marker='o')
ax2.plot(k_indices[zoom_mask], pgd_norm[zoom_mask], label="PGD-20 (n=1,000)", linewidth=2, linestyle='--', marker='x')
ax2.set_title("Verification: Formant Transition Region (k=10 to 20)")
ax2.set_xlabel("Quefrency Index (k)")
ax2.set_ylabel("Normalized Ratio")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("large_sample_verification.png", dpi=300)
print("Verification complete. Check large_sample_verification.png.")

# Optional: Save plot to Drive as well
def save_to_drive(local_path, remote_name=None):
    """Save plot to Google Drive project folder."""
    ensure_project_dir()
    remote_name = remote_name or os.path.basename(local_path)
    dest_path = os.path.join(PROJECT_DIR, remote_name)
    try:
        shutil.copy2(local_path, dest_path)
        print(f"  [DRIVE OK] {local_path}  →  {dest_path}")
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_path}: {e}")

save_to_drive("large_sample_verification.png")
print("Plot saved to Google Drive project folder.")
os.sync

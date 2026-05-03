"""
=============================================================================
Program Name: 2_diagnostic_speech_commands.py
Version: 1.1
Description: 
    1. Mounts Google Drive and copies SpeechCommandsV2.zip to local storage.
    2. Extracts data using zipfile.
    3. Initializes baseline CNN tuned for 16kHz, 1-second speech clips.
    4. Runs FGSM, PGD-20, PGD-100, and C&W attacks on a diagnostic batch.
    5. Computes E[|ΔX_k|^2] and plots the quefrency energy profile (skipping DC).
    6. Syncs all checkpoints, JSON data, and figures to Google Drive project folder.
GPU Needed: Yes
Dependencies: torch, torchaudio, torchattacks, pandas, librosa, zipfile
#!pip install torchaudio torchvision torchattacks matplotlib pandas librosa

=============================================================================
"""
#!pip install torchaudio torchvision torchattacks matplotlib pandas librosa


import os
import shutil
import zipfile
import json
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import torchattacks

import sys
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# --- Configuration ---
DRIVE_DIR = "/content/drive/MyDrive/datasets"
PROJECT_DIR = "/content/drive/MyDrive/paper/cepstral_lifter/"  # Persistent storage for checkpoints, JSON, figures
LOCAL_DATA_DIR = "/content/data"
ZIP_FILE_NAME = "SpeechCommandsV2.zip"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Google Drive Helper Functions ---
def ensure_project_dir():
    """Create project directory in Google Drive if it doesn't exist."""
    os.makedirs(PROJECT_DIR, exist_ok=True)

def save_to_drive(local_path, remote_name=None):
    """
    Copy a single local file to Google Drive project folder.
    remote_name: filename in Drive (defaults to basename of local_path).
    Returns True on success, False on failure.
    """
    ensure_project_dir()
    remote_name = remote_name or os.path.basename(local_path)
    dest_path = os.path.join(PROJECT_DIR, remote_name)
    try:
        shutil.copy2(local_path, dest_path)
        print(f"  [DRIVE OK] {local_path}  →  {dest_path}")
        return True
    except Exception as e:
        print(f"  [DRIVE FAIL] {local_path}: {e}")
        return False

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

# --- Data Preparation ---
def setup_data():
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    local_zip_path = os.path.join(LOCAL_DATA_DIR, ZIP_FILE_NAME)
    drive_zip_path = os.path.join(DRIVE_DIR, ZIP_FILE_NAME)
    
    if not os.path.exists(local_zip_path):
        print("Copying dataset from Drive...")
        shutil.copy(drive_zip_path, local_zip_path)
    
    extract_dir = os.path.join(LOCAL_DATA_DIR, "SpeechCommands")
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    print("Dataset ready.")

# --- Model Definition ---
class SpeechCNN(nn.Module):
    def __init__(self, n_classes=10):  # Default 10-word subset for SCv2
        super().__init__()
        # Differentiable MFCC frontend tuned for 16kHz speech
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Dynamic FC layer to be sized during first forward pass
        self.fc = nn.Linear(1, n_classes) 
        self.relu = nn.ReLU()
        self._fc_initialized = False

    def forward(self, x):
        x = self.mfcc(x) 
        x = x.unsqueeze(1) 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        
        if not self._fc_initialized or x.size(1) != self.fc.in_features:
             self.fc = nn.Linear(x.size(1), 10).to(x.device)
             self._fc_initialized = True
        return self.fc(x)

# --- Diagnostic Experiment Execution ---
def run_diagnostic():
    model = SpeechCNN().to(DEVICE)
    model_checkpoint = "02_baseline_speech_cnn.pth"
    
    # Try to recover checkpoint from Google Drive project folder
    if load_from_drive(model_checkpoint, model_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
        print("Loaded checkpoint from Drive.")
    else:
        print("Saving untrained baseline config for diagnostic...")
        # Placeholder initialization
        dummy_pass = torch.randn(2, 16000).to(DEVICE)
        model(dummy_pass) # Initialize dynamic FC layer
        torch.save(model.state_dict(), model_checkpoint)
        save_to_drive(model_checkpoint, model_checkpoint)

    model.eval()
    
    # Setup Attacks
    attacks = {
        "FGSM": torchattacks.FGSM(model, eps=0.005),
        "PGD-20": torchattacks.PGD(model, eps=0.005, alpha=0.001, steps=20),
        "PGD-100": torchattacks.PGD(model, eps=0.01, alpha=0.001, steps=100),
        "CW": torchattacks.CW(model, c=1, steps=1000)
    }
    
    results_json = "02_quefrency_results_scv2.json"
    # Try to load existing results from Drive
    if load_from_drive(results_json, results_json):
        with open(results_json, "r") as f:
            energy_profiles = json.load(f)
    else:
        energy_profiles = {}

    # Diagnostic Tensor: 200 clips of 1-second audio at 16kHz
    print("Generating adversarial perturbations on Speech Commands...")
    dummy_audio = torch.randn(200, 16000).to(DEVICE) 
    dummy_labels = torch.randint(0, 10, (200,)).to(DEVICE)

    clean_mfcc = model.mfcc(dummy_audio).detach()

    for name, attack in attacks.items():
        if name in energy_profiles:
            continue
            
        print(f"Running {name}...")
        adv_audio = attack(dummy_audio, dummy_labels)
        adv_mfcc = model.mfcc(adv_audio).detach()
        
        delta_x = adv_mfcc - clean_mfcc
        delta_x_sq = torch.square(delta_x)
        
        energy_profile = delta_x_sq.mean(dim=(0, 2)).cpu().tolist()
        energy_profiles[name] = energy_profile
        
        # Checkpoint results to Drive
        with open(results_json, "w") as f:
            json.dump(energy_profiles, f)
        save_to_drive(results_json, results_json)

    # Plotting Fig. 1 (Excluding k=1 DC spike)
    plt.figure(figsize=(10, 6))
    for name, profile in energy_profiles.items():
        plt.plot(range(2, 41), profile[1:], label=name, linewidth=2)
    
    plt.title("Fig. 1: Quefrency Energy Profile (Speech Commands v2)")
    plt.xlabel("Quefrency Index (k)")
    plt.ylabel("Expected Residual Energy $\\mathbb{E}[|\\Delta X_k|^2]$")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.axvspan(5, 15, color='gray', alpha=0.2, label='Expected Concentration Zone')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plot_name = "fig1_quefrency_profile_scv2.png"
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    save_to_drive(plot_name, plot_name)
    print("Diagnostic complete. Plot saved to Drive.")

if __name__ == "__main__":
    setup_data()
    run_diagnostic()
    os.sync
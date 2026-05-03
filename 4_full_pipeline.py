"""
=============================================================================
Program Name: 4_full_pipeline.py
Version: 1.5
Change log:
  v1.0 - Unified Stage 1 + Stage 2 pipeline.
  v1.1 - Fixed RAM crash: on-disk Dataset.
  v1.2 - Fixed AttributeError: separated class definitions.
  v1.3 - Fixed sanity gap: identity lifter init, single model object.
  v1.4 - Added resume capability for Stage 2.
  v1.5 - Recalibrated epsilon. EPS_MFCC 0.15 -> 0.74 (5% of norm range).
          ALPHA_MFCC updated to 0.185 (EPS/4).
          S2 outputs renamed 10v15_* to avoid collision with v1.4 files.
          S1 checkpoint reused from Google Drive (unchanged).
          Resume detects 10v15_checkpoint_epXX.pth only (not old v1.4).
          Checkpoints saved every 3 epochs
Description:
    Single self-contained pipeline to train a learnable cepstral lifter (L-Sinc) 
    for adversarial robustness on Speech Commands v2, covering Stage 1 clean 
    pre-training and Stage 2 PGD adversarial training with automatic session-resume 
    capability.
    Upload this file to Colab. Run once. Resumes automatically if  session disconnects.
GPU: T4 or better. Runtime: ~90-120 min full, less on resume.
Dependencies: torch, torchaudio, numpy, matplotlib, tqdm
=============================================================================
"""

import sys, os, json, shutil, zipfile, collections, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE          = torch.device("cuda")
BATCH_SIZE      = 256
S1_EPOCHS       = 15
S2_EPOCHS       = 27
S1_LR           = 1e-3
NORM_BATCHES    = 100
CLIPS_PER_CLASS = 1500
SEED            = 42
N_MFCC          = 40
N_CLASSES       = 35
SAMPLE_RATE     = 16000
CLIP_SAMPLES    = 16000
NUM_WORKERS     = 2

LAMBDA_LOW      = 0.5
LAMBDA_HIGH     = 0.1
EPS_MFCC        = 0.74        # 5% of normalised MFCC range ~14.7
ALPHA_MFCC      = 0.185       # EPS / 4

LOCAL_DATA_DIR  = "./data"
DRIVE_ZIP       = "/content/drive/MyDrive/datasets/SpeechCommandsV2.zip"
PROJECT_DIR     = "/content/drive/MyDrive/paper/cepstral_lifter/"  # Persistent storage
SENTINEL_WORD   = "yes"

# File names (same as before, now stored in Google Drive)
S1_CKPT        = "04_stage1_checkpoint.pth"
S2_CKPT        = "04_defended_model.pth"
LIFTER_NPY     = "04_lifter_history.npy"
S1_CURVE_PNG   = "04_stage1_curve.png"
S2_CURVE_PNG   = "04_stage2_curve.png"
FIG3_PNG       = "04_fig3_lifter_profiles.png"
S1_STATS_JSON  = "04_stage1_stats.json"
S2_STATS_JSON  = "04_stage2_stats.json"
S2_CKPT_PREFIX = "04_checkpoint_ep"

torch.manual_seed(SEED)
random.seed(SEED)

# =============================================================================
# GOOGLE DRIVE HELPER FUNCTIONS
# =============================================================================
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

def save_to_drive_many(paths):
    """
    Upload a list of local file paths to Google Drive.
    paths: list of str  OR  list of (local_path, remote_name) tuples.
    """
    for item in paths:
        if isinstance(item, (list, tuple)):
            save_to_drive(item[0], item[1])
        else:
            save_to_drive(item)

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

def list_drive_files():
    """List files in the Google Drive project directory."""
    ensure_project_dir()
    try:
        return [f for f in os.listdir(PROJECT_DIR) if os.path.isfile(os.path.join(PROJECT_DIR, f))]
    except Exception as e:
        print(f"  [DRIVE] Could not list files: {e}")
        return []

def find_latest_s2_checkpoint():
    """Find the latest Stage 2 checkpoint in Google Drive."""
    files = list_drive_files()
    ckpts = [f for f in files
             if f.startswith(S2_CKPT_PREFIX) and f.endswith(".pth")]
    if not ckpts:
        return 0, None
    epochs = []
    for f in ckpts:
        try:
            ep = int(f.replace(S2_CKPT_PREFIX, "").replace(".pth", ""))
            epochs.append((ep, f))
        except ValueError:
            continue
    if not epochs:
        return 0, None
    epochs.sort(key=lambda x: x[0], reverse=True)
    return epochs[0]

# =============================================================================
# 0. GPU CHECK
# =============================================================================
if not torch.cuda.is_available():
    print("[ERROR] GPU not detected. Switch to T4 GPU runtime.")
    sys.exit(1)
print("=" * 60)
print(f"CUDA available: True  |  v1.6  |  EPS_MFCC={EPS_MFCC}")
print("=" * 60 + "\n")

# =============================================================================
# 1. DETERMINE RUN MODE
# =============================================================================
print("Checking Google Drive for existing checkpoints...")
drive_files      = list_drive_files()
s1_on_drive      = S1_CKPT in drive_files
resume_ep, resume_ckpt_name = find_latest_s2_checkpoint()

print(f"  Stage 1 on Drive : {'YES — ' + S1_CKPT if s1_on_drive else 'NO'}")
print(f"  Latest v1.6 S2 : {resume_ckpt_name + ' (ep'+str(resume_ep)+')' if resume_ckpt_name else 'NONE'}")

old_v14 = [f for f in drive_files
           if f.startswith("10_checkpoint_ep") and f.endswith(".pth")]
if old_v14:
    print(f"  Old v1.4 files : {len(old_v14)} found — will NOT be loaded.")

RESUME_S2 = s1_on_drive and resume_ep > 0
SKIP_S1   = s1_on_drive
print(f"\nMode: {'RESUME S2 ep'+str(resume_ep) if RESUME_S2 else 'SKIP S1, S2 ep1' if SKIP_S1 else 'FULL PIPELINE'}\n")

# =============================================================================
# 2. DATASET SETUP
# =============================================================================
sentinel = os.path.join(LOCAL_DATA_DIR, SENTINEL_WORD)
if os.path.isdir(sentinel):
    n = len([d for d in os.listdir(LOCAL_DATA_DIR)
             if os.path.isdir(os.path.join(LOCAL_DATA_DIR, d))
             and not d.startswith("_")])
    print(f"[Dataset] Found locally ({n} word folders).\n")
else:
    print("[Dataset] Not found locally — mounting Drive...")
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("  Drive mounted.")
    except Exception as e:
        print(f"  [ERROR] {e}"); sys.exit(1)
    if not os.path.isfile(DRIVE_ZIP):
        print(f"  [ERROR] {DRIVE_ZIP} not found."); sys.exit(1)
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    zip_dst = os.path.join(LOCAL_DATA_DIR, "SpeechCommandsV2.zip")
    print(f"  Copying zip ({os.path.getsize(DRIVE_ZIP)/1e6:.0f} MB)...")
    shutil.copy2(DRIVE_ZIP, zip_dst)
    print("  Unzipping...")
    with zipfile.ZipFile(zip_dst, "r") as zf:
        zf.extractall(LOCAL_DATA_DIR)
    os.remove(zip_dst)
    if not os.path.isdir(sentinel):
        print(f"  [ERROR] '{SENTINEL_WORD}' missing after unzip."); sys.exit(1)
    print("  Dataset ready.\n")

# =============================================================================
# 3. FILE-PATH INDEX
# =============================================================================
print(f"Building file-path index ({CLIPS_PER_CLASS}/class x {N_CLASSES})...")
WORD_FOLDERS = sorted([
    d for d in os.listdir(LOCAL_DATA_DIR)
    if os.path.isdir(os.path.join(LOCAL_DATA_DIR, d))
    and not d.startswith("_")
])
KNOWN_WORDS = [
    "backward","bed","bird","cat","dog","down","eight","five",
    "follow","forward","four","go","happy","house","learn","left",
    "marvin","nine","no","off","on","one","right","seven","sheila",
    "six","stop","three","tree","two","up","visual","wow","yes","zero"
]
WORD_FOLDERS = [w for w in WORD_FOLDERS if w in KNOWN_WORDS]
assert len(WORD_FOLDERS) == N_CLASSES, \
    f"Expected {N_CLASSES} word folders, got {len(WORD_FOLDERS)}. " \
    f"Missing: {set(KNOWN_WORDS) - set(WORD_FOLDERS)}"

label_to_idx = {w: i for i, w in enumerate(WORD_FOLDERS)}
idx_to_label = {i: w for w, i in label_to_idx.items()}

all_paths, all_labels = [], []
for word in WORD_FOLDERS:
    word_dir, cnt = os.path.join(LOCAL_DATA_DIR, word), 0
    for fname in sorted(os.listdir(word_dir)):
        if not fname.endswith(".wav"): continue
        all_paths.append(os.path.join(word_dir, fname))
        all_labels.append(label_to_idx[word])
        cnt += 1
        if cnt >= CLIPS_PER_CLASS: break

cc = collections.Counter(all_labels)
assert len(cc) == N_CLASSES
print(f"Index: {len(all_paths):,} files | "
      f"min={min(cc.values())} max={max(cc.values())}\n")

# =============================================================================
# 4. ON-DISK DATASET
# =============================================================================
class SpeechCommandsDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths; self.labels = labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        wf, sr = torchaudio.load(self.paths[idx])
        if sr != SAMPLE_RATE:
            wf = torchaudio.functional.resample(wf, sr, SAMPLE_RATE)
        wf = wf.squeeze(0)
        if wf.shape[0] < CLIP_SAMPLES:
            wf = F.pad(wf, (0, CLIP_SAMPLES - wf.shape[0]))
        else:
            wf = wf[:CLIP_SAMPLES]
        return wf, self.labels[idx]

full_dataset = SpeechCommandsDataset(all_paths, all_labels)

# =============================================================================
# 5. TRAIN/VAL SPLIT
# =============================================================================
n_total = len(full_dataset)
val_sz  = int(0.1 * n_total)
indices = list(range(n_total))
random.Random(SEED).shuffle(indices)
train_indices = indices[:n_total - val_sz]
val_indices   = indices[n_total - val_sz:]

train_ds = Subset(full_dataset, train_indices)
val_ds   = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}\n")

# =============================================================================
# 6. MFCC TRANSFORM
# =============================================================================
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
).to(DEVICE)

# =============================================================================
# 7. MFCC NORMALISER
# =============================================================================
class MFCCNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_MFCC))
        self.register_buffer("std",  torch.ones(N_MFCC))

    def fit(self, loader, n_batches=100):
        print(f"Fitting MFCCNormalizer over {n_batches} batches...")
        self.eval()
        accum = []
        for i, (audio, _) in enumerate(loader):
            with torch.no_grad():
                m = mfcc_transform(audio.to(DEVICE))
                accum.append(m.permute(1,0,2).reshape(N_MFCC,-1).cpu())
            if i >= n_batches - 1: break
        all_m     = torch.cat(accum, dim=1)
        self.mean = all_m.mean(1).to(DEVICE)
        self.std  = all_m.std(1).clamp(min=1e-5).to(DEVICE)
        print(f"  Mean [{self.mean.min():.2f}, {self.mean.max():.2f}]  "
              f"Std [{self.std.min():.2f}, {self.std.max():.2f}]")

    def forward(self, x):
        return (x - self.mean.view(1,-1,1)) / self.std.view(1,-1,1)


normalizer = MFCCNormalizer().to(DEVICE)
normalizer.fit(train_loader, NORM_BATCHES)

with torch.no_grad():
    dx, _ = next(iter(train_loader))
    nm_check = normalizer(mfcc_transform(dx.to(DEVICE)))
nr = nm_check.max().item() - nm_check.min().item()
print(f"Post-norm [{nm_check.min():.3f}, {nm_check.max():.3f}]  "
      f"std={nm_check.std():.4f}")
print(f"EPS_MFCC={EPS_MFCC} = {100*EPS_MFCC/nr:.1f}% of range "
      f"(target 3-8%)\n")

# =============================================================================
# 8. MODEL CLASSES
# =============================================================================
class LearnableLifter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("phi", torch.full((N_MFCC,), 2.0))
        lam = torch.ones(N_MFCC, dtype=torch.float32)
        lam[1:15] = LAMBDA_LOW
        lam[17:]  = LAMBDA_HIGH
        self.register_buffer("lambda_weight", lam)
        self.w = nn.Parameter(torch.zeros(N_MFCC))

    def get_ell(self): return torch.sigmoid(self.w) * self.phi

    def forward(self, mfcc):
        return mfcc * self.get_ell().view(1,-1,1)

    def get_lifter_values(self):
        with torch.no_grad(): return self.get_ell().cpu()

    def get_reg_loss(self):
        return torch.sum((self.get_ell()-1.0)**2 * self.lambda_weight)


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32,  3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,  3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(7680, N_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(self.drop(x.view(x.size(0),-1)))


class DefendedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = normalizer
        self.lifter     = LearnableLifter()
        self.classifier = ClassifierHead()

    def from_norm_mfcc(self, nm):
        return self.classifier(self.lifter(nm))

    def forward(self, waveform):
        return self.from_norm_mfcc(
            self.normalizer(mfcc_transform(waveform))
        )

# =============================================================================
# 9. PGD ATTACK
# =============================================================================
def pgd_mfcc(model_fn, nm_clean, lbls, eps, alpha, steps):
    adv = nm_clean.detach().clone()
    adv = adv + torch.zeros_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, nm_clean - eps, nm_clean + eps)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        F.cross_entropy(model_fn(adv), lbls).backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.clamp(adv, nm_clean - eps, nm_clean + eps)
    return adv.detach()

# =============================================================================
# 10. EVAL HELPER
# =============================================================================
def eval_clean(model, forward_fn, loader):
    model.eval()
    corr, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            nm   = normalizer(mfcc_transform(x))
            corr += (forward_fn(nm).argmax(1)==y).sum().item()
            n    += y.size(0)
    model.train()
    return 100.0 * corr / n

# =============================================================================
# 11. BUILD MODEL
# =============================================================================
model = DefendedModel().to(DEVICE)
with torch.no_grad():
    ell_i = model.lifter.get_lifter_values()
print(f"Lifter at init: min={ell_i.min():.6f} max={ell_i.max():.6f} "
      f"(identity=1.000000)\n")
criterion = nn.CrossEntropyLoss()

# =============================================================================
# 12. STAGE 1  (skipped if S1 checkpoint on Google Drive)
# =============================================================================
s1_final_val = None
s1_history   = {"epoch":[], "ce":[], "train_acc":[], "val_acc":[]}

if SKIP_S1:
    print("=" * 60)
    print("STAGE 1: Skipped — loading from Google Drive")
    print("=" * 60 + "\n")
    if not os.path.isfile(S1_CKPT):
        ok = load_from_drive(S1_CKPT, S1_CKPT)
        if not ok:
            print(f"[ERROR] Cannot download {S1_CKPT} from Drive."); sys.exit(1)
    ckpt_s1 = torch.load(S1_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt_s1["model_state_dict"], strict=False)
    model.normalizer.mean = ckpt_s1["normalizer_mean"].to(DEVICE)
    model.normalizer.std  = ckpt_s1["normalizer_std"].to(DEVICE)
    normalizer.mean       = model.normalizer.mean
    normalizer.std        = model.normalizer.std
    s1_final_val          = ckpt_s1["final_val_acc"]
    print(f"  Loaded. Stage 1 final val acc: {s1_final_val:.2f}%\n")

else:
    print("=" * 60)
    print(f"STAGE 1: Clean Pre-Training ({S1_EPOCHS} epochs)")
    print("=" * 60 + "\n")

    opt1   = optim.Adam([
        {"params": model.classifier.parameters(), "lr": S1_LR},
        {"params": model.lifter.parameters(),     "lr": S1_LR * 0.01},
        {"params": model.normalizer.parameters(), "lr": 0.0},
    ])
    sched1 = optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=S1_EPOCHS, eta_min=1e-5)

    for epoch in range(1, S1_EPOCHS + 1):
        model.train()
        sum_loss, corr, total_n = 0.0, 0, 0
        for audio, lbls in tqdm(train_loader,
                                 desc=f"S1 Ep{epoch:02d}", leave=False):
            audio, lbls = audio.to(DEVICE), lbls.to(DEVICE)
            with torch.no_grad():
                nm = normalizer(mfcc_transform(audio))
            opt1.zero_grad()
            out  = model.from_norm_mfcc(nm)
            loss = criterion(out, lbls)
            loss.backward(); opt1.step()
            sum_loss += loss.item()
            corr     += (out.argmax(1)==lbls).sum().item()
            total_n  += lbls.size(0)
        sched1.step()

        vacc = eval_clean(model, model.from_norm_mfcc, val_loader)
        ce   = sum_loss / len(train_loader)
        tacc = 100.0 * corr / total_n
        s1_history["epoch"].append(epoch)
        s1_history["ce"].append(ce)
        s1_history["train_acc"].append(tacc)
        s1_history["val_acc"].append(vacc)
        print(f"Ep{epoch:02d} | CE:{ce:.4f} | Train:{tacc:.2f}% | "
              f"Val:{vacc:.2f}%")

    s1_final_val = s1_history["val_acc"][-1]
    print(f"\nStage 1 complete. Final Val Acc: {s1_final_val:.2f}%\n")

    torch.save({
        "model_state_dict" : model.state_dict(),
        "normalizer_mean"  : normalizer.mean.cpu(),
        "normalizer_std"   : normalizer.std.cpu(),
        "label_to_idx"     : label_to_idx,
        "idx_to_label"     : idx_to_label,
        "word_folders"     : WORD_FOLDERS,
        "train_indices"    : train_indices,
        "val_indices"      : val_indices,
        "n_classes"        : N_CLASSES,
        "n_mfcc"           : N_MFCC,
        "final_val_acc"    : s1_final_val,
    }, S1_CKPT)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(s1_history["epoch"], s1_history["ce"], "b-o", lw=2)
    axes[0].set_title("Stage 1: CE Loss"); axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].plot(s1_history["epoch"], s1_history["train_acc"],
                 "b-o", lw=2, label="Train")
    axes[1].plot(s1_history["epoch"], s1_history["val_acc"],
                 "r--s", lw=2, label="Val")
    axes[1].set_title("Stage 1: Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(S1_CURVE_PNG, dpi=150); plt.close()
    with open(S1_STATS_JSON, "w") as f:
        json.dump(s1_history, f, indent=2)
    print("Saving Stage 1 outputs to Google Drive...")
    save_to_drive_many([S1_CKPT, S1_CURVE_PNG, S1_STATS_JSON])
    print("Stage 1 Drive save complete.\n")

# =============================================================================
# 13. SANITY CHECK
# =============================================================================
sanity = eval_clean(model, model.from_norm_mfcc, val_loader)
print(f"Sanity check — Val Clean Acc: {sanity:.2f}%  "
      f"(Stage 1: {s1_final_val:.2f}%)")
assert abs(sanity - s1_final_val) < 3.0, \
    f"Sanity FAILED: {sanity:.2f}% vs {s1_final_val:.2f}%"
print("Sanity check PASSED.\n")

# =============================================================================
# 14. STAGE 2 SETUP
# =============================================================================
print("=" * 60)
print(f"STAGE 2: Adversarial Training ({S2_EPOCHS} epochs)  "
      f"EPS={EPS_MFCC}  ALPHA={ALPHA_MFCC}")
print("=" * 60 + "\n")

opt2 = optim.Adam([
    {"params": model.classifier.parameters(), "lr": 1e-5},
    {"params": model.lifter.parameters(),     "lr": 1e-4},
    {"params": model.normalizer.parameters(), "lr": 0.0},
])
sched2 = optim.lr_scheduler.CosineAnnealingLR(
    opt2, T_max=S2_EPOCHS, eta_min=1e-6)

s2_history = {
    "epoch":[], "ce":[], "adv_ce":[], "reg":[],
    "train_clean_acc":[], "train_adv_acc":[],
    "val_clean_acc":[], "val_adv_acc":{}
}
lifter_history = {}

if RESUME_S2:
    print(f"Resuming from epoch {resume_ep} checkpoint...")
    if not os.path.isfile(resume_ckpt_name):
        ok = load_from_drive(resume_ckpt_name, resume_ckpt_name)
        if not ok:
            print(f"[ERROR] Cannot download {resume_ckpt_name} from Drive"); sys.exit(1)
    model.load_state_dict(
        torch.load(resume_ckpt_name, map_location=DEVICE))
    print(f"  Weights loaded.")

    if not os.path.isfile(S2_STATS_JSON):
        load_from_drive(S2_STATS_JSON, S2_STATS_JSON)
    if os.path.isfile(S2_STATS_JSON):
        with open(S2_STATS_JSON) as f:
            s2_history = json.load(f)
        s2_history["val_adv_acc"] = {
            int(k): v for k, v in s2_history["val_adv_acc"].items()
        }
        print(f"  S2 history loaded ({len(s2_history['epoch'])} epochs).")

    if not os.path.isfile(LIFTER_NPY):
        load_from_drive(LIFTER_NPY, LIFTER_NPY)
    if os.path.isfile(LIFTER_NPY):
        lifter_history = dict(
            np.load(LIFTER_NPY, allow_pickle=True).item())
        print(f"  Lifter history loaded ({len(lifter_history)} epochs).")

    for _ in range(resume_ep):
        sched2.step()
    print(f"  Scheduler at epoch {resume_ep}.\n")

start_epoch = resume_ep + 1
identity_np = np.ones(N_MFCC)
print(f"Training epoch {start_epoch} → {S2_EPOCHS}.\n")

# =============================================================================
# 15. STAGE 2 TRAINING LOOP
# =============================================================================
for epoch in range(start_epoch, S2_EPOCHS + 1):
    model.train()
    pgd_steps = 3 if epoch <= 5 else 7

    sum_ce, sum_adv, sum_reg      = 0.0, 0.0, 0.0
    clean_corr, adv_corr, total_n = 0, 0, 0

    for audio, lbls in tqdm(train_loader,
                             desc=f"S2 Ep{epoch:02d} PGD-{pgd_steps}",
                             leave=False):
        audio, lbls = audio.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            nm = normalizer(mfcc_transform(audio))

        opt2.zero_grad()
        clean_out = model.from_norm_mfcc(nm)
        ce_loss   = criterion(clean_out, lbls)
        with torch.no_grad():
            sum_ce     += ce_loss.item()
            clean_corr += (clean_out.argmax(1)==lbls).sum().item()
            total_n    += lbls.size(0)

        model.eval()
        nm_adv = pgd_mfcc(model.from_norm_mfcc, nm,
                           lbls, EPS_MFCC, ALPHA_MFCC, pgd_steps)
        model.train()

        opt2.zero_grad()
        adv_out     = model.from_norm_mfcc(nm_adv)
        adv_ce_loss = criterion(adv_out, lbls)
        reg_loss    = model.lifter.get_reg_loss()
        with torch.no_grad():
            sum_adv  += adv_ce_loss.item()
            sum_reg  += reg_loss.item()
            adv_corr += (adv_out.argmax(1)==lbls).sum().item()

        total_loss = ce_loss + 0.5 * adv_ce_loss + reg_loss
        total_loss.backward()
        opt2.step()

    sched2.step()
    vacc = eval_clean(model, model.from_norm_mfcc, val_loader)

    val_adv_str = ""
    if epoch % 3 == 0:
        model.eval()
        vac, vn = 0, 0
        for vx, vy in val_loader:
            vx, vy = vx.to(DEVICE), vy.to(DEVICE)
            nm_v   = normalizer(mfcc_transform(vx))
            nm_adv = pgd_mfcc(model.from_norm_mfcc, nm_v,
                               vy, EPS_MFCC, ALPHA_MFCC, steps=20)
            vac += (model.from_norm_mfcc(nm_adv)
                    .argmax(1)==vy).sum().item()
            vn  += vy.size(0)
        model.train()
        vaa = 100.0 * vac / vn
        s2_history["val_adv_acc"][epoch] = vaa
        val_adv_str = f" | ValAdv(PGD-20):{vaa:.2f}%"

        mid = f"{S2_CKPT_PREFIX}{epoch:02d}.pth"
        torch.save(model.state_dict(), mid)
        save_to_drive(mid)
        with open(S2_STATS_JSON, "w") as f:
            json.dump(s2_history, f, indent=2, default=str)
        save_to_drive(S2_STATS_JSON)
        np.save(LIFTER_NPY, lifter_history)
        save_to_drive(LIFTER_NPY)
        print(f"  [DRIVE] ep{epoch:02d} saved.")

    ell        = model.lifter.get_lifter_values()
    vuln_drift = (ell[17:] - 1.0).abs().mean().item()
    lifter_history[epoch] = ell.numpy()

    nb    = len(train_loader)
    ce_e  = sum_ce/nb; adv_e = sum_adv/nb; reg_e = sum_reg/nb
    tacc  = 100.0 * clean_corr / total_n
    tadv  = 100.0 * adv_corr   / total_n

    s2_history["epoch"].append(epoch)
    s2_history["ce"].append(ce_e)
    s2_history["adv_ce"].append(adv_e)
    s2_history["reg"].append(reg_e)
    s2_history["train_clean_acc"].append(tacc)
    s2_history["train_adv_acc"].append(tadv)
    s2_history["val_clean_acc"].append(vacc)

    print(f"Ep{epoch:02d} | CE:{ce_e:.4f} | AdvCE:{adv_e:.4f} | "
          f"Reg:{reg_e:.4f} | TrnClean:{tacc:.2f}% | "
          f"TrnAdv:{tadv:.2f}% | ValClean:{vacc:.2f}% | "
          f"VulnDrift:{vuln_drift:.4f}" + val_adv_str)

# =============================================================================
# 16. SAVE ALL OUTPUTS
# =============================================================================
torch.save({
    "model_state_dict"  : model.state_dict(),
    "normalizer_mean"   : normalizer.mean.cpu(),
    "normalizer_std"    : normalizer.std.cpu(),
    "label_to_idx"      : label_to_idx,
    "idx_to_label"      : idx_to_label,
    "word_folders"      : WORD_FOLDERS,
    "train_indices"     : train_indices,
    "val_indices"       : val_indices,
    "n_classes"         : N_CLASSES,
    "n_mfcc"            : N_MFCC,
    "s1_final_val_acc"  : s1_final_val,
    "s2_history"        : s2_history,
    "eps_mfcc"          : EPS_MFCC,
    "alpha_mfcc"        : ALPHA_MFCC,
    "lambda_low"        : LAMBDA_LOW,
    "lambda_high"       : LAMBDA_HIGH,
}, S2_CKPT)

np.save(LIFTER_NPY, lifter_history)
with open(S2_STATS_JSON, "w") as f:
    json.dump(s2_history, f, indent=2, default=str)

# =============================================================================
# 17. PLOTS
# =============================================================================
ep = s2_history["epoch"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(ep, s2_history["ce"],     "b-o", lw=2, label="Clean CE")
axes[0].plot(ep, s2_history["adv_ce"], "r-s", lw=2, label="Adv CE")
axes[0].set_title(f"Stage 2 Loss (ε={EPS_MFCC})")
axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[1].plot(ep, s2_history["train_clean_acc"], "b-o",  lw=2, label="TrnClean")
axes[1].plot(ep, s2_history["val_clean_acc"],   "b--s", lw=2, label="ValClean")
axes[1].plot(ep, s2_history["train_adv_acc"],   "r-o",  lw=2, label="TrnAdv")
axes[1].set_title("Stage 2 Accuracy")
axes[1].set_xlabel("Epoch"); axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[2].plot(ep, s2_history["reg"], "g-o", lw=2)
axes[2].set_title("Regularisation Loss")
axes[2].set_xlabel("Epoch")
axes[2].grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(); plt.savefig(S2_CURVE_PNG, dpi=150); plt.close()

k_axis    = np.arange(1, N_MFCC + 1)
ell_final = lifter_history.get(
    S2_EPOCHS, lifter_history[max(lifter_history.keys())])
rho_k  = np.linspace(0.05, 1.2, N_MFCC)
wiener = 1.0 / (1.0 + rho_k)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap = plt.cm.Reds(np.linspace(0.3, 1.0, len(lifter_history)))
for i, (ep_i, ell_i) in enumerate(sorted(lifter_history.items())):
    axes[0].plot(k_axis, ell_i, color=cmap[i], alpha=0.5, linewidth=1)
axes[0].plot(k_axis, identity_np, "k--", lw=2,
             label=r"Identity $\ell_k=1$")
axes[0].plot(k_axis, ell_final, "r-", lw=2.5,
             label=f"Learned ep{max(lifter_history.keys())}")
axes[0].axvspan(2,  15, alpha=0.12, color="green", label="Formant Band")
axes[0].axvspan(18, 40, alpha=0.12, color="red",   label="Vulnerability Band")
axes[0].set_title("Fig. 3a: Lifter Evolution")
axes[0].set_xlabel("Quefrency Index (k)")
axes[0].set_ylabel(r"Lifter Output $\ell_k$")
axes[0].legend(fontsize=8); axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(k_axis, identity_np, "k--", lw=2, label=r"Identity $\ell_k=1$")
axes[1].plot(k_axis, wiener,      "b:",  lw=2, label=r"Wiener Optimal")
axes[1].plot(k_axis, ell_final,   "r-",  lw=2.5, label=r"Learned $\ell_k(\theta)$")
axes[1].axvspan(2,  15, alpha=0.12, color="green", label="Formant Band")
axes[1].axvspan(18, 40, alpha=0.12, color="red",   label="Vulnerability Band")
axes[1].set_title("Fig. 3b: Learned vs Identity vs Wiener")
axes[1].set_xlabel("Quefrency Index (k)")
axes[1].set_ylabel(r"Lifter Output $\ell_k$")
axes[1].legend(fontsize=8); axes[1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout(); plt.savefig(FIG3_PNG, dpi=150); plt.close()

# =============================================================================
# 18. FINAL DRIVE UPLOAD
# =============================================================================
print("\nSaving all final outputs to Google Drive...")
save_to_drive_many([S2_CKPT, LIFTER_NPY, S2_CURVE_PNG, FIG3_PNG, S2_STATS_JSON])

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"  EPS_MFCC              : {EPS_MFCC}")
print(f"  Stage 1 final val acc : {s1_final_val:.2f}%")
print(f"  Stage 2 final val acc : {s2_history['val_clean_acc'][-1]:.2f}%")
if s2_history["val_adv_acc"]:
    last_ep  = max(int(k) for k in s2_history["val_adv_acc"])
    last_acc = s2_history["val_adv_acc"][last_ep]
    print(f"  Stage 2 val adv acc   : {last_acc:.2f}% "
          f"(PGD-20, ep{last_ep})")
print("=" * 60)
os.sync
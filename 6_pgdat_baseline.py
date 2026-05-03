"""
=============================================================================
Program Name: 6_pgdat_baseline
Version: 1.1
Description:
    Trains a PGD-AT baseline: same architecture and data as 04_full_pipeline,
    but WITHOUT the L-Sinc lifter. Standard adversarial training with PGD
    at EPS_MFCC=0.15 (same threat model as defended model).

    This is Table 1 baseline M3 (PGD-AT baseline).

    Pipeline:
      1. Dataset setup (Drive mount if needed)
      2. Same file-path index, split, normaliser as 04_full_pipeline
      3. Stage 1: clean pre-training 15 epochs (loaded from Drive if available)
      4. Stage 2: PGD-AT 27 epochs, NO lifter, EPS=0.15
      5. Save to Google Drive

    Outputs (all saved to Google Drive):
      06_pgdat_model.pth
      06_pgdat_stage2_curve.png
      06_pgdat_stage2_stats.json
      06_pgdat_checkpoint_epXX.pth  (every 3 epochs)

GPU: T4 or better. Runtime: ~60-75 min (S1 skipped, S2 only).
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
# CONFIGURATION — identical to 04_full_pipeline except no lifter
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

EPS_MFCC        = 0.15        # same threat model as defended model
ALPHA_MFCC      = 0.0375      # EPS / 4

PROJECT_DIR    = "/content/drive/MyDrive/paper/cepstral_lifter/"  # Persistent storage
LOCAL_DATA_DIR = "./data"
DRIVE_ZIP      = "/content/drive/MyDrive/datasets/SpeechCommandsV2.zip"
SENTINEL_WORD  = "yes"

S1_CKPT        = "04_stage1_checkpoint.pth"   # reuse from pipeline
S2_CKPT        = "06_pgdat_model.pth"
S2_CURVE_PNG   = "06_pgdat_stage2_curve.png"
S2_STATS_JSON  = "06_pgdat_stage2_stats.json"
S2_CKPT_PREFIX = "06_pgdat_checkpoint_ep"

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

def find_latest_checkpoint():
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
    print("[ERROR] GPU not detected."); sys.exit(1)
print("=" * 60)
print(f"CUDA available: True  |  PGD-AT Baseline  |  EPS={EPS_MFCC}")
print("=" * 60 + "\n")

# =============================================================================
# 1. DETERMINE RUN MODE
# =============================================================================
print("Checking Google Drive for existing checkpoints...")
drive_files   = list_drive_files()
s1_on_drive   = S1_CKPT in drive_files
resume_ep, resume_ckpt_name = find_latest_checkpoint()

print(f"  Stage 1 on Drive  : {'YES' if s1_on_drive else 'NO'}")
print(f"  Latest PGD-AT     : {resume_ckpt_name or 'NONE'}")

RESUME_S2 = resume_ep > 0
SKIP_S1   = s1_on_drive
print(f"\nMode: {'RESUME ep'+str(resume_ep) if RESUME_S2 else 'SKIP S1, S2 ep1' if SKIP_S1 else 'FULL'}\n")

# =============================================================================
# 2. DATASET
# =============================================================================
sentinel = os.path.join(LOCAL_DATA_DIR, SENTINEL_WORD)
if os.path.isdir(sentinel):
    n = len([d for d in os.listdir(LOCAL_DATA_DIR)
             if os.path.isdir(os.path.join(LOCAL_DATA_DIR, d))
             and not d.startswith("_")])
    print(f"[Dataset] Found locally ({n} word folders).\n")
else:
    print("[Dataset] Not found — mounting Drive...")
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
    with zipfile.ZipFile(zip_dst, "r") as zf:
        zf.extractall(LOCAL_DATA_DIR)
    os.remove(zip_dst)
    if not os.path.isdir(sentinel):
        print(f"  [ERROR] '{SENTINEL_WORD}' missing."); sys.exit(1)
    print("  Dataset ready.\n")

# =============================================================================
# 3. FILE-PATH INDEX  (identical ordering to 04_full_pipeline)
# =============================================================================
print(f"Building file-path index ({CLIPS_PER_CLASS}/class x {N_CLASSES})...")
WORD_FOLDERS = sorted([
    d for d in os.listdir(LOCAL_DATA_DIR)
    if os.path.isdir(os.path.join(LOCAL_DATA_DIR, d))
    and not d.startswith("_")
])
assert len(WORD_FOLDERS) == N_CLASSES
label_to_idx = {w: i for i, w in enumerate(WORD_FOLDERS)}

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
# 5. TRAIN/VAL SPLIT  (identical SEED to 04_full_pipeline)
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
print(f"EPS_MFCC={EPS_MFCC} = {100*EPS_MFCC/nr:.1f}% of range\n")

# =============================================================================
# 8. MODEL — ClassifierHead only, NO lifter
# =============================================================================
class ClassifierHead(nn.Module):
    """Identical to 04_full_pipeline ClassifierHead."""
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


class PGDATModel(nn.Module):
    """Normalizer -> Classifier. No lifter."""
    def __init__(self):
        super().__init__()
        self.normalizer = normalizer
        self.classifier = ClassifierHead()

    def from_norm_mfcc(self, nm):
        return self.classifier(nm)

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
def eval_clean(model, loader):
    model.eval()
    corr, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            nm   = normalizer(mfcc_transform(x))
            corr += (model.from_norm_mfcc(nm).argmax(1)==y).sum().item()
            n    += y.size(0)
    model.train()
    return 100.0 * corr / n

# =============================================================================
# 11. BUILD MODEL + LOAD S1 WEIGHTS
# =============================================================================
model     = PGDATModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
s1_final_val = None

print("=" * 60)
print("STAGE 1: Loading S1 weights from Google Drive")
print("=" * 60 + "\n")

if not os.path.isfile(S1_CKPT):
    ok = load_from_drive(S1_CKPT, S1_CKPT)
    if not ok:
        print(f"[ERROR] Cannot download {S1_CKPT} from Drive."); sys.exit(1)

ckpt_s1 = torch.load(S1_CKPT, map_location=DEVICE)

# Extract classifier weights from full DefendedModel state dict
s1_state     = ckpt_s1["model_state_dict"]
s1_clf_state = {k.replace("classifier.", ""): v
                for k, v in s1_state.items()
                if k.startswith("classifier.")}
s1_norm_mean = ckpt_s1["normalizer_mean"].to(DEVICE)
s1_norm_std  = ckpt_s1["normalizer_std"].to(DEVICE)

assert len(s1_clf_state) > 0, "No classifier keys found in S1 checkpoint."
model.classifier.load_state_dict(s1_clf_state, strict=True)
model.normalizer.mean = s1_norm_mean
model.normalizer.std  = s1_norm_std
normalizer.mean       = s1_norm_mean
normalizer.std        = s1_norm_std
s1_final_val          = ckpt_s1["final_val_acc"]

print(f"  S1 classifier weights loaded ({len(s1_clf_state)} keys).")
print(f"  Normaliser loaded from S1 checkpoint.")
print(f"  S1 final val acc: {s1_final_val:.2f}%\n")

# Sanity check
sanity = eval_clean(model, val_loader)
print(f"Sanity check — Val Clean Acc: {sanity:.2f}%  "
      f"(S1: {s1_final_val:.2f}%)")
assert abs(sanity - s1_final_val) < 3.0, \
    f"Sanity FAILED: {sanity:.2f}% vs {s1_final_val:.2f}%"
print("Sanity check PASSED.\n")

# =============================================================================
# 12. STAGE 2 SETUP
# =============================================================================
print("=" * 60)
print(f"STAGE 2: PGD-AT (no lifter)  {S2_EPOCHS} epochs  "
      f"EPS={EPS_MFCC}  ALPHA={ALPHA_MFCC}")
print("=" * 60 + "\n")

opt2   = optim.Adam(model.parameters(), lr=1e-5,
                    weight_decay=1e-4)
sched2 = optim.lr_scheduler.CosineAnnealingLR(
    opt2, T_max=S2_EPOCHS, eta_min=1e-6)

s2_history = {
    "epoch":[], "ce":[], "adv_ce":[], "reg":[],
    "train_clean_acc":[], "train_adv_acc":[],
    "val_clean_acc":[], "val_adv_acc":{}
}

if RESUME_S2:
    print(f"Resuming from epoch {resume_ep}...")
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

    for _ in range(resume_ep):
        sched2.step()
    print(f"  Scheduler at epoch {resume_ep}.\n")

start_epoch = resume_ep + 1
print(f"Training epoch {start_epoch} → {S2_EPOCHS}.\n")

# =============================================================================
# 13. STAGE 2 TRAINING LOOP
# =============================================================================
for epoch in range(start_epoch, S2_EPOCHS + 1):
    model.train()
    pgd_steps = 3 if epoch <= 5 else 7

    sum_ce, sum_adv               = 0.0, 0.0
    clean_corr, adv_corr, total_n = 0, 0, 0

    for audio, lbls in tqdm(train_loader,
                             desc=f"Ep{epoch:02d} PGD-{pgd_steps}",
                             leave=False):
        audio, lbls = audio.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            nm = normalizer(mfcc_transform(audio))

        # Clean pass
        opt2.zero_grad()
        clean_out = model.from_norm_mfcc(nm)
        ce_loss   = criterion(clean_out, lbls)
        with torch.no_grad():
            sum_ce     += ce_loss.item()
            clean_corr += (clean_out.argmax(1)==lbls).sum().item()
            total_n    += lbls.size(0)

        # PGD attack
        model.eval()
        nm_adv = pgd_mfcc(model.from_norm_mfcc, nm,
                           lbls, EPS_MFCC, ALPHA_MFCC, pgd_steps)
        model.train()

        # Adversarial pass
        opt2.zero_grad()
        adv_out     = model.from_norm_mfcc(nm_adv)
        adv_ce_loss = criterion(adv_out, lbls)
        with torch.no_grad():
            sum_adv  += adv_ce_loss.item()
            adv_corr += (adv_out.argmax(1)==lbls).sum().item()

        total_loss = ce_loss + 0.5 * adv_ce_loss
        total_loss.backward()
        opt2.step()

    sched2.step()
    vacc = eval_clean(model, val_loader)

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
        print(f"  [DRIVE] ep{epoch:02d} saved.")

    nb    = len(train_loader)
    ce_e  = sum_ce  / nb
    adv_e = sum_adv / nb
    tacc  = 100.0 * clean_corr / total_n
    tadv  = 100.0 * adv_corr   / total_n

    s2_history["epoch"].append(epoch)
    s2_history["ce"].append(ce_e)
    s2_history["adv_ce"].append(adv_e)
    s2_history["reg"].append(0.0)
    s2_history["train_clean_acc"].append(tacc)
    s2_history["train_adv_acc"].append(tadv)
    s2_history["val_clean_acc"].append(vacc)

    print(f"Ep{epoch:02d} | CE:{ce_e:.4f} | AdvCE:{adv_e:.4f} | "
          f"TrnClean:{tacc:.2f}% | TrnAdv:{tadv:.2f}% | "
          f"ValClean:{vacc:.2f}%" + val_adv_str)

# =============================================================================
# 14. SAVE ALL OUTPUTS
# =============================================================================
torch.save({
    "model_state_dict"  : model.state_dict(),
    "normalizer_mean"   : normalizer.mean.cpu(),
    "normalizer_std"    : normalizer.std.cpu(),
    "label_to_idx"      : label_to_idx,
    "word_folders"      : WORD_FOLDERS,
    "n_classes"         : N_CLASSES,
    "n_mfcc"            : N_MFCC,
    "s1_final_val_acc"  : s1_final_val,
    "s2_history"        : s2_history,
    "eps_mfcc"          : EPS_MFCC,
    "has_lifter"        : False,
}, S2_CKPT)

with open(S2_STATS_JSON, "w") as f:
    json.dump(s2_history, f, indent=2, default=str)

# =============================================================================
# 15. TRAINING CURVE PLOT
# =============================================================================
ep = s2_history["epoch"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ep, s2_history["ce"],     "b-o", lw=2, label="Clean CE")
axes[0].plot(ep, s2_history["adv_ce"], "r-s", lw=2, label="Adv CE")
axes[0].set_title(f"PGD-AT Baseline: Loss (ε={EPS_MFCC})")
axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(ep, s2_history["train_clean_acc"], "b-o",  lw=2, label="TrnClean")
axes[1].plot(ep, s2_history["val_clean_acc"],   "b--s", lw=2, label="ValClean")
axes[1].plot(ep, s2_history["train_adv_acc"],   "r-o",  lw=2, label="TrnAdv")
axes[1].set_title("PGD-AT Baseline: Accuracy")
axes[1].set_xlabel("Epoch"); axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(S2_CURVE_PNG, dpi=150)
plt.close()

# =============================================================================
# 16. SAVE TO GOOGLE DRIVE
# =============================================================================
print("\nSaving all outputs to Google Drive...")
save_to_drive_many([S2_CKPT, S2_CURVE_PNG, S2_STATS_JSON])

print("\n" + "=" * 60)
print("PGD-AT BASELINE COMPLETE")
print(f"  EPS_MFCC              : {EPS_MFCC}")
print(f"  Stage 1 final val acc : {s1_final_val:.2f}%")
print(f"  Stage 2 final val acc : {s2_history['val_clean_acc'][-1]:.2f}%")
if s2_history["val_adv_acc"]:
    last_ep  = max(int(k) for k in s2_history["val_adv_acc"])
    last_acc = s2_history["val_adv_acc"][last_ep]
    print(f"  Val adv acc (PGD-20)  : {last_acc:.2f}% (ep{last_ep})")
print("=" * 60)
os.sync
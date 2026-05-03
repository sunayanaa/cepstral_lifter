"""
=============================================================================
Program Name: 5b_evaluation_table1.py
Version: 1.1
Description:
    Produces Table 1b of the paper: Clean and Robust Accuracy across
    attack types and defense configurations.
    Variant of 05_evaluation_table1.py with M3 replaced by PGD-AT baseline.

    Models evaluated:
      M1: Undefended baseline (S1 classifier, no lifter, no adv training)
      M2: Fixed classical sinusoidal lifter (L=22), S1 classifier,
          no adversarial training
      M3: PGD-AT baseline (06_pgdat_model.pth) — same architecture as M1,
          adversarially trained with PGD at EPS=0.15, no lifter

    Attacks evaluated (normalised MFCC space):
      - Clean (no attack)
      - FGSM   (eps=EPS_MFCC=0.15, 1 step)
      - PGD-20 (eps=EPS_MFCC=0.15, 20 steps, alpha=EPS/4)
      - PGD-100(eps=EPS_MFCC=0.15, 100 steps, alpha=EPS/4)
      - C&W-L2 (50 steps, lr=0.01, confidence=0)

    Epsilon: EPS_MFCC=0.15 only (single threat model, matching training).

    Checkpoints (stored in Google Drive):
      S1_CKPT = 04_stage1_checkpoint.pth  (M1, M2 weights)
      S2_CKPT = 06_pgdat_model.pth        (M3 PGD-AT weights)

    Outputs (saved to Google Drive):
      - 05b_table1.json
      - 05b_table1.txt
      - 05b_table1.png

GPU: Yes. Runtime: ~20-30 min.
Dependencies: torch, torchaudio, numpy, matplotlib, tqdm
=============================================================================
"""

import sys, os, json, collections, random, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
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
BATCH_SIZE      = 128
N_MFCC          = 40
N_CLASSES       = 35
SAMPLE_RATE     = 16000
CLIP_SAMPLES    = 16000
NUM_WORKERS     = 2
CLIPS_PER_CLASS = 1500
SEED            = 42

EPS_MFCC   = 0.15
ALPHA_FRAC = 0.25

PROJECT_DIR    = "/content/drive/MyDrive/paper/cepstral_lifter/"  # Persistent storage
LOCAL_DATA_DIR = "./data"
SENTINEL_WORD  = "yes"
S1_CKPT        = "04_stage1_checkpoint.pth"
S2_CKPT        = "06_pgdat_model.pth"

OUT_JSON = "05b_table1.json"
OUT_TXT  = "05b_table1.txt"
OUT_PNG  = "05b_table1.png"

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

# =============================================================================
# 0. GPU CHECK
# =============================================================================
if not torch.cuda.is_available():
    print("[ERROR] GPU not detected."); sys.exit(1)
print("=" * 60)
print("CUDA available: True — Evaluation Script 05b")
print("=" * 60 + "\n")

# =============================================================================
# 1. DATASET
# =============================================================================
sentinel = os.path.join(LOCAL_DATA_DIR, SENTINEL_WORD)
if not os.path.isdir(sentinel):
    print("[Dataset] Not found — mounting Drive...")
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        print(f"[ERROR] {e}"); sys.exit(1)
    import zipfile
    zip_src = "/content/drive/MyDrive/datasets/SpeechCommandsV2.zip"
    zip_dst = os.path.join(LOCAL_DATA_DIR, "SpeechCommandsV2.zip")
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    print("  Copying zip...")
    shutil.copy2(zip_src, zip_dst)
    with zipfile.ZipFile(zip_dst, "r") as zf:
        zf.extractall(LOCAL_DATA_DIR)
    os.remove(zip_dst)
    print("  Dataset ready.")
else:
    print("[Dataset] Found locally.\n")

# =============================================================================
# 2. FILE-PATH INDEX + SPLIT  (identical to training)
# =============================================================================
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

class SpeechDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths; self.labels = labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        wf, sr = torchaudio.load(self.paths[idx])
        if sr != SAMPLE_RATE:
            wf = torchaudio.functional.resample(wf, sr, SAMPLE_RATE)
        wf = wf.squeeze(0)
        wf = F.pad(wf, (0, max(0, CLIP_SAMPLES - wf.shape[0])))[:CLIP_SAMPLES]
        return wf, self.labels[idx]

full_ds     = SpeechDataset(all_paths, all_labels)
n_total     = len(full_ds)
indices     = list(range(n_total))
random.Random(SEED).shuffle(indices)
val_indices = indices[n_total - int(0.1 * n_total):]
val_ds      = Subset(full_ds, val_indices)
val_loader  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
print(f"Evaluation set: {len(val_ds):,} clips\n")

# =============================================================================
# 3. MFCC TRANSFORM
# =============================================================================
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
).to(DEVICE)

# =============================================================================
# 4. MODEL COMPONENTS
# =============================================================================
class MFCCNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_MFCC))
        self.register_buffer("std",  torch.ones(N_MFCC))
    def forward(self, x):
        return (x - self.mean.view(1,-1,1)) / self.std.view(1,-1,1)


class ClassicalLifter(nn.Module):
    """Fixed sinusoidal lifter L=22 normalised to mean=1."""
    def __init__(self, L=22):
        super().__init__()
        k   = torch.arange(1, N_MFCC+1, dtype=torch.float32)
        phi = 1.0 + (L/2.0) * torch.sin(torch.pi * k / L)
        phi = phi / phi.mean()
        self.register_buffer("phi", phi)
    def forward(self, x):
        return x * self.phi.view(1,-1,1)


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32,  3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,  3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.0)
        self.fc    = nn.Linear(7680, N_CLASSES)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(self.drop(x.view(x.size(0),-1)))


class UndefendedModel(nn.Module):
    """normalizer -> classifier (no lifter)."""
    def __init__(self, normalizer, classifier):
        super().__init__()
        self.normalizer = normalizer
        self.classifier = classifier
    def from_norm_mfcc(self, nm): return self.classifier(nm)
    def forward(self, wf):
        return self.from_norm_mfcc(self.normalizer(mfcc_transform(wf)))


class ClassicalModel(nn.Module):
    """normalizer -> classical lifter -> classifier."""
    def __init__(self, normalizer, lifter, classifier):
        super().__init__()
        self.normalizer = normalizer
        self.lifter     = lifter
        self.classifier = classifier
    def from_norm_mfcc(self, nm): return self.classifier(self.lifter(nm))
    def forward(self, wf):
        return self.from_norm_mfcc(self.normalizer(mfcc_transform(wf)))

# =============================================================================
# 5. DOWNLOAD CHECKPOINTS FROM GOOGLE DRIVE
# =============================================================================
for ckpt_name in [S1_CKPT, S2_CKPT]:
    if not os.path.isfile(ckpt_name):
        ok = load_from_drive(ckpt_name, ckpt_name)
        if not ok:
            print(f"[ERROR] {ckpt_name} not available in Drive."); sys.exit(1)

ckpt_s1 = torch.load(S1_CKPT, map_location=DEVICE)
ckpt_s2 = torch.load(S2_CKPT, map_location=DEVICE)

print(f"S1 checkpoint keys: {list(ckpt_s1.keys())}")
print(f"S2 checkpoint keys: {list(ckpt_s2.keys())}\n")

# =============================================================================
# 6. EXTRACT WEIGHTS
#    Both checkpoints store full DefendedModel / PGDATModel state dicts.
#    Keys are prefixed: classifier.*, normalizer.*, (lifter.* in S1 only)
# =============================================================================
def extract_substate(state_dict, prefix):
    return {k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)}


def build_normalizer(state_dict):
    """
    Build MFCCNormalizer from a model state dict.
    Handles both prefixed keys (normalizer.mean) and
    direct checkpoint fields (normalizer_mean).
    """
    n = MFCCNormalizer().to(DEVICE)
    # Try prefixed keys first (from model state dict)
    norm_sub = extract_substate(state_dict, "normalizer.")
    if "mean" in norm_sub and "std" in norm_sub:
        n.mean = norm_sub["mean"].to(DEVICE)
        n.std  = norm_sub["std"].to(DEVICE)
    else:
        raise KeyError("Cannot find normalizer.mean/std in state dict.")
    return n


# S1 weights
s1_state     = ckpt_s1.get("model_state_dict", ckpt_s1)
s1_clf_state = extract_substate(s1_state, "classifier.")
norm_m1_obj  = build_normalizer(s1_state)
norm_m2_obj  = build_normalizer(s1_state)

assert len(s1_clf_state) > 0, "No classifier keys in S1 checkpoint."
print(f"S1 classifier keys: {len(s1_clf_state)}")

# S2 weights (PGD-AT — PGDATModel: normalizer.* + classifier.*, no lifter)
s2_state     = ckpt_s2.get("model_state_dict", ckpt_s2)
s2_clf_state = extract_substate(s2_state, "classifier.")
norm_m3_obj  = build_normalizer(s2_state)

assert len(s2_clf_state) > 0, "No classifier keys in S2 checkpoint."
print(f"S2 classifier keys: {len(s2_clf_state)}\n")

# =============================================================================
# 7. BUILD THREE MODELS
# =============================================================================
# M1 — Undefended (S1 weights, no lifter, no adv training)
clf_m1 = ClassifierHead().to(DEVICE)
clf_m1.load_state_dict(s1_clf_state, strict=True)
m1 = UndefendedModel(norm_m1_obj, clf_m1).to(DEVICE)

# M2 — Classical sinusoidal lifter (S1 weights, no adv training)
clf_m2  = ClassifierHead().to(DEVICE)
clf_m2.load_state_dict(s1_clf_state, strict=True)
lift_m2 = ClassicalLifter().to(DEVICE)
m2 = ClassicalModel(norm_m2_obj, lift_m2, clf_m2).to(DEVICE)

# M3 — PGD-AT baseline (S2 weights, no lifter)
clf_m3 = ClassifierHead().to(DEVICE)
clf_m3.load_state_dict(s2_clf_state, strict=True)
m3 = UndefendedModel(norm_m3_obj, clf_m3).to(DEVICE)

models = {
    "M1-Undefended"  : m1,
    "M2-ClassLifter" : m2,
    "M3-PGD-AT"      : m3,
}

# Pre-evaluation clean accuracy check
print("Pre-evaluation clean accuracy check:")
for mname, mdl in models.items():
    mdl.eval()
    corr, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            nm   = mdl.normalizer(mfcc_transform(x))
            corr += (mdl.from_norm_mfcc(nm).argmax(1)==y).sum().item()
            n    += y.size(0)
    print(f"  {mname:<24}: {100*corr/n:.2f}%")
print()

# =============================================================================
# 8. ATTACK FUNCTIONS
# =============================================================================
def fgsm(model_fn, nm, y, eps):
    adv = nm.detach().clone().requires_grad_(True)
    F.cross_entropy(model_fn(adv), y).backward()
    with torch.no_grad():
        adv = nm + eps * adv.grad.sign()
    return adv.detach()

def pgd(model_fn, nm, y, eps, alpha, steps):
    adv = nm.detach().clone()
    adv = adv + torch.zeros_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, nm - eps, nm + eps)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        F.cross_entropy(model_fn(adv), y).backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.clamp(adv, nm - eps, nm + eps)
    return adv.detach()

def cw_l2(model_fn, nm, y, steps=50, lr=0.01, confidence=0):
    delta = torch.zeros_like(nm, requires_grad=True)
    opt   = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        perturbed = nm + delta
        logits    = model_fn(perturbed)
        B         = y.size(0)
        correct   = logits[range(B), y]
        mask      = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1e9)
        other     = (logits - mask).max(1).values
        clf_loss  = F.relu(correct - other + confidence).mean()
        l2_loss   = (delta ** 2).mean()
        loss      = clf_loss + 0.01 * l2_loss
        opt.zero_grad(); loss.backward(); opt.step()
    return (nm + delta).detach()

# =============================================================================
# 9. EVALUATION FUNCTION
# =============================================================================
def evaluate_model(model, loader, eps, desc=""):
    model.eval()
    alpha   = eps * ALPHA_FRAC
    results = {k: [0, 0] for k in ["clean","fgsm","pgd20","pgd100","cw"]}

    for audio, y in tqdm(loader, desc=f"  {desc}", leave=False):
        audio, y = audio.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            nm = model.normalizer(mfcc_transform(audio))
        fn = model.from_norm_mfcc

        with torch.no_grad():
            results["clean"][0] += (fn(nm).argmax(1)==y).sum().item()
            results["clean"][1] += y.size(0)

        adv = fgsm(fn, nm, y, eps)
        with torch.no_grad():
            results["fgsm"][0] += (fn(adv).argmax(1)==y).sum().item()
            results["fgsm"][1] += y.size(0)

        adv = pgd(fn, nm, y, eps, alpha, steps=20)
        with torch.no_grad():
            results["pgd20"][0] += (fn(adv).argmax(1)==y).sum().item()
            results["pgd20"][1] += y.size(0)

        adv = pgd(fn, nm, y, eps, alpha, steps=100)
        with torch.no_grad():
            results["pgd100"][0] += (fn(adv).argmax(1)==y).sum().item()
            results["pgd100"][1] += y.size(0)

        adv = cw_l2(fn, nm, y)
        with torch.no_grad():
            results["cw"][0] += (fn(adv).argmax(1)==y).sum().item()
            results["cw"][1] += y.size(0)

    return {k: 100.0 * v[0] / v[1] for k, v in results.items()}

# =============================================================================
# 10. RUN EVALUATION
# =============================================================================
attacks    = ["clean", "fgsm", "pgd20", "pgd100", "cw"]
atk_labels = {"clean":"Clean","fgsm":"FGSM","pgd20":"PGD-20",
               "pgd100":"PGD-100","cw":"C&W-L2"}

print(f"\n{'='*60}")
print(f"Evaluating at EPS={EPS_MFCC}")
print(f"{'='*60}")

all_results = {}
for mname, mdl in models.items():
    print(f"\n  Model: {mname}")
    res = evaluate_model(mdl, val_loader, EPS_MFCC, desc=mname[:25])
    all_results[mname] = res
    print(f"  Clean:{res['clean']:.2f}%  FGSM:{res['fgsm']:.2f}%  "
          f"PGD-20:{res['pgd20']:.2f}%  PGD-100:{res['pgd100']:.2f}%  "
          f"C&W:{res['cw']:.2f}%")

# =============================================================================
# 11. FORMAT TABLE
# =============================================================================
def format_table():
    lines = []
    lines.append(f"\nTable 1b: Accuracy (%) under adversarial attacks "
                 f"[ε={EPS_MFCC}, normalised MFCC space]")
    hdr = f"{'Model':<24}" + "".join(f"{atk_labels[a]:>10}" for a in attacks)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for mname in models:
        res = all_results[mname]
        row = f"{mname:<24}" + "".join(f"{res[a]:>10.2f}" for a in attacks)
        lines.append(row)
    lines.append("-" * len(hdr))
    m1_pgd = all_results["M1-Undefended"]["pgd20"]
    m3_pgd = all_results["M3-PGD-AT"]["pgd20"]
    lines.append(f"\nRobustness gain M3 vs M1 (PGD-20): {m3_pgd-m1_pgd:+.2f}%")
    return "\n".join(lines)

table_str = format_table()
print("\n" + table_str)

with open(OUT_TXT, "w") as f: f.write(table_str)
with open(OUT_JSON, "w") as f: json.dump(all_results, f, indent=2)

# =============================================================================
# 12. PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"M1-Undefended":"#d62728",
          "M2-ClassLifter":"#ff7f0e",
          "M3-PGD-AT":"#1f77b4"}
x     = np.arange(len(attacks))
width = 0.25

for i, mname in enumerate(models):
    vals = [all_results[mname][a] for a in attacks]
    bars = ax.bar(x + (i-1)*width, vals, width,
                  label=mname, color=colors[mname], alpha=0.85)
    for bar, val in zip(bars, vals):
        if val > 1.0:
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([atk_labels[a] for a in attacks])
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Table 1b: Robustness Evaluation (ε={EPS_MFCC}, normalised MFCC)")
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"\nPlot saved: {OUT_PNG}")

# =============================================================================
# 13. SAVE TO GOOGLE DRIVE
# =============================================================================
print("\nSaving outputs to Google Drive...")
save_to_drive_many([OUT_JSON, OUT_TXT, OUT_PNG])
print("Evaluation complete.")
os.sync

"""
=============================================================================
Program Name: 5_evaluation_table1.py
Version: 1.1
Description:
    Produces Table 1 of the paper: Clean and Robust Accuracy across
    attack types and defense configurations.

    Models evaluated:
      M1: Undefended baseline (no lifter, no adversarial training)
      M2: Fixed classical lifter (sinusoidal, L=22), no adv training
      M3: L-Sinc defended model (04_defended_model.pth) — our method

    Attacks evaluated:
      - Clean (no attack)
      - FGSM  (eps=EPS_MFCC, 1 step)
      - PGD-20 (eps=EPS_MFCC, 20 steps)
      - PGD-100 (eps=EPS_MFCC, 100 steps)
      - CW-L2  (binary search, 10 steps, confidence=0)

    All attacks operate in normalised MFCC space.
    Epsilon matched to defended model training: EPS_MFCC=0.15.
    Also evaluated at 5x: EPS_MFCC=0.74 to show calibrated results.

    Outputs:
      - 05_table1.json  : full results
      - 05_table1.txt   : formatted ASCII table (paste into paper)
      - 05_table1.png   : publication-quality figure
    All saved to Google Drive project folder.
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
BATCH_SIZE      = 128          # smaller for attack stability
N_MFCC          = 40
N_CLASSES       = 35
SAMPLE_RATE     = 16000
CLIP_SAMPLES    = 16000
NUM_WORKERS     = 2
CLIPS_PER_CLASS = 1500
SEED            = 42

# Epsilon values to evaluate
EPS_TRAIN  = 0.15              # epsilon used during adv training
EPS_CALIB  = 0.74              # calibrated 5% of norm range (~correct scale)
ALPHA_FRAC = 0.25              # alpha = eps * ALPHA_FRAC

PROJECT_DIR    = "/content/drive/MyDrive/paper/cepstral_lifter/"  # Persistent storage
LOCAL_DATA_DIR = "./data"
SENTINEL_WORD  = "yes"
S1_CKPT        = "04_stage1_checkpoint.pth"
S2_CKPT        = "04_defended_model.pth"

OUT_JSON = "05_table1.json"
OUT_TXT  = "05_table1.txt"
OUT_PNG  = "05_table1.png"

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
print("CUDA available: True — Evaluation Script")
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
    print(f"  Copying zip...")
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

full_ds = SpeechDataset(all_paths, all_labels)
n_total = len(full_ds)
indices = list(range(n_total))
random.Random(SEED).shuffle(indices)
val_indices = indices[n_total - int(0.1 * n_total):]

# Use only validation set for evaluation
val_ds     = Subset(full_ds, val_indices)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
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
    """Fixed sinusoidal lifter L=22 — baseline M2."""
    def __init__(self, L=22):
        super().__init__()
        k   = torch.arange(1, N_MFCC+1, dtype=torch.float32)
        phi = 1.0 + (L/2.0) * torch.sin(torch.pi * k / L)
        # Normalise so mean=1 to avoid feature scale shift
        phi = phi / phi.mean()
        self.register_buffer("phi", phi)
    def forward(self, x):
        return x * self.phi.view(1,-1,1)


class LearnableLifter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("phi", torch.full((N_MFCC,), 2.0))
        self.w = nn.Parameter(torch.zeros(N_MFCC))
    def forward(self, x):
        return x * (torch.sigmoid(self.w) * self.phi).view(1,-1,1)


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32,  3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,  3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.0)    # eval mode — dropout off anyway
        self.fc    = nn.Linear(7680, N_CLASSES)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(self.drop(x.view(x.size(0),-1)))


# =============================================================================
# 5. DOWNLOAD CHECKPOINTS IF NEEDED
# =============================================================================
for ckpt in [S1_CKPT, S2_CKPT]:
    if not os.path.isfile(ckpt):
        load_from_drive(ckpt, ckpt)
    if not os.path.isfile(ckpt):
        print(f"[ERROR] {ckpt} not available in Drive."); sys.exit(1)

# Load S1 checkpoint (for normaliser stats + undefended weights)
ckpt_s1 = torch.load(S1_CKPT, map_location=DEVICE)
ckpt_s2 = torch.load(S2_CKPT, map_location=DEVICE)

# =============================================================================
# 6. BUILD THREE MODELS
# =============================================================================
def make_normalizer(ckpt):
    n = MFCCNormalizer().to(DEVICE)
    n.mean = ckpt["normalizer_mean"].to(DEVICE)
    n.std  = ckpt["normalizer_std"].to(DEVICE)
    return n

# M1: Undefended — S1 classifier, identity lifter (no lifter)
class UndefendedModel(nn.Module):
    def __init__(self, normalizer, classifier):
        super().__init__()
        self.normalizer = normalizer
        self.classifier = classifier
    def from_norm_mfcc(self, nm): return self.classifier(nm)
    def forward(self, wf):
        return self.from_norm_mfcc(self.normalizer(mfcc_transform(wf)))

# M2: Fixed sinusoidal lifter — S1 classifier + classical lifter
class ClassicalModel(nn.Module):
    def __init__(self, normalizer, lifter, classifier):
        super().__init__()
        self.normalizer = normalizer
        self.lifter     = lifter
        self.classifier = classifier
    def from_norm_mfcc(self, nm): return self.classifier(self.lifter(nm))
    def forward(self, wf):
        return self.from_norm_mfcc(self.normalizer(mfcc_transform(wf)))

# M3: Defended — full S2 model
class DefendedModel(nn.Module):
    def __init__(self, normalizer, lifter, classifier):
        super().__init__()
        self.normalizer = normalizer
        self.lifter     = lifter
        self.classifier = classifier
    def from_norm_mfcc(self, nm): return self.classifier(self.lifter(nm))
    def forward(self, wf):
        return self.from_norm_mfcc(self.normalizer(mfcc_transform(wf)))

# Instantiate M1
norm_m1  = make_normalizer(ckpt_s1)
clf_m1   = ClassifierHead().to(DEVICE)
clf_m1.load_state_dict(ckpt_s1["model_state_dict"], strict=False)
m1 = UndefendedModel(norm_m1, clf_m1).to(DEVICE)

# Instantiate M2 (same S1 weights, add classical lifter)
norm_m2  = make_normalizer(ckpt_s1)
clf_m2   = ClassifierHead().to(DEVICE)
clf_m2.load_state_dict(ckpt_s1["model_state_dict"], strict=False)
lift_m2  = ClassicalLifter().to(DEVICE)
m2 = ClassicalModel(norm_m2, lift_m2, clf_m2).to(DEVICE)

# Instantiate M3
norm_m3  = make_normalizer(ckpt_s2)
lift_m3  = LearnableLifter().to(DEVICE)
clf_m3   = ClassifierHead().to(DEVICE)

# Load S2 weights — map keys
s2_state = ckpt_s2["model_state_dict"]
norm_state = {k.replace("normalizer.", ""): v
              for k, v in s2_state.items() if k.startswith("normalizer.")}
lift_state = {k.replace("lifter.", ""): v
              for k, v in s2_state.items() if k.startswith("lifter.")}
clf_state  = {k.replace("classifier.", ""): v
              for k, v in s2_state.items() if k.startswith("classifier.")}
lift_m3.load_state_dict(lift_state, strict=False)
clf_m3.load_state_dict(clf_state,   strict=True)
m3 = DefendedModel(norm_m3, lift_m3, clf_m3).to(DEVICE)

models = {
    "M1-Undefended"      : m1,
    "M2-ClassicalLifter" : m2,
    "M3-LSinc (Ours)"    : m3,
}
print("Models built:")
for name in models: print(f"  {name}")
print()

# =============================================================================
# 7. ATTACK FUNCTIONS
# =============================================================================
def fgsm(model_fn, nm, y, eps):
    nm_adv = nm.detach().clone().requires_grad_(True)
    F.cross_entropy(model_fn(nm_adv), y).backward()
    return (nm + eps * nm_adv.grad.sign()).detach()

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
    """Simplified C&W L2 in MFCC space."""
    adv   = nm.detach().clone()
    delta = torch.zeros_like(adv, requires_grad=True)
    opt   = torch.optim.Adam([delta], lr=lr)
    best  = adv.clone()

    for _ in range(steps):
        perturbed = nm + delta
        logits    = model_fn(perturbed)
        # C&W loss: misclassify + L2 penalty
        B         = y.size(0)
        correct   = logits[range(B), y]
        # max logit of other classes
        mask      = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1e9)
        other     = (logits - mask).max(1).values
        clf_loss  = F.relu(correct - other + confidence).mean()
        l2_loss   = (delta ** 2).sum(dim=(1,2,3)).mean() \
                    if delta.dim() == 4 \
                    else (delta ** 2).sum(dim=(1,2)).mean()
        loss = clf_loss + 0.01 * l2_loss
        opt.zero_grad(); loss.backward(); opt.step()
        best = (nm + delta).detach()

    return best.detach()

# =============================================================================
# 8. EVALUATION FUNCTION
# =============================================================================
def evaluate_model(model, loader, eps, desc=""):
    """
    Returns dict: clean, fgsm, pgd20, pgd100, cw accuracies (%).
    """
    model.eval()
    normalizer = model.normalizer

    results = {k: [0, 0] for k in
               ["clean", "fgsm", "pgd20", "pgd100", "cw"]}
    alpha = eps * ALPHA_FRAC

    for audio, y in tqdm(loader, desc=desc, leave=False):
        audio, y = audio.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            nm = normalizer(mfcc_transform(audio))

        model_fn = model.from_norm_mfcc

        # Clean
        with torch.no_grad():
            pred = model_fn(nm).argmax(1)
        results["clean"][0] += (pred == y).sum().item()
        results["clean"][1] += y.size(0)

        # FGSM
        nm_adv = fgsm(model_fn, nm, y, eps)
        with torch.no_grad():
            pred = model_fn(nm_adv).argmax(1)
        results["fgsm"][0] += (pred == y).sum().item()
        results["fgsm"][1] += y.size(0)

        # PGD-20
        nm_adv = pgd(model_fn, nm, y, eps, alpha, steps=20)
        with torch.no_grad():
            pred = model_fn(nm_adv).argmax(1)
        results["pgd20"][0] += (pred == y).sum().item()
        results["pgd20"][1] += y.size(0)

        # PGD-100
        nm_adv = pgd(model_fn, nm, y, eps, alpha, steps=100)
        with torch.no_grad():
            pred = model_fn(nm_adv).argmax(1)
        results["pgd100"][0] += (pred == y).sum().item()
        results["pgd100"][1] += y.size(0)

        # C&W
        nm_adv = cw_l2(model_fn, nm, y)
        with torch.no_grad():
            pred = model_fn(nm_adv).argmax(1)
        results["cw"][0] += (pred == y).sum().item()
        results["cw"][1] += y.size(0)

    return {k: 100.0 * v[0] / v[1] for k, v in results.items()}

# =============================================================================
# 9. RUN EVALUATION — both epsilon values
# =============================================================================
all_results = {}

for eps_label, eps_val in [("eps_train", EPS_TRAIN),
                            ("eps_calib", EPS_CALIB)]:
    print(f"\n{'='*60}")
    print(f"Evaluating at EPS={eps_val:.2f} ({eps_label})")
    print(f"{'='*60}")
    all_results[eps_label] = {}
    for mname, mdl in models.items():
        print(f"\n  Model: {mname}")
        res = evaluate_model(
            mdl, val_loader, eps_val,
            desc=f"  {mname[:20]:20s} eps={eps_val:.2f}"
        )
        all_results[eps_label][mname] = res
        print(f"  Clean:{res['clean']:.2f}%  FGSM:{res['fgsm']:.2f}%  "
              f"PGD-20:{res['pgd20']:.2f}%  PGD-100:{res['pgd100']:.2f}%  "
              f"C&W:{res['cw']:.2f}%")

# =============================================================================
# 10. FORMAT TABLE 1 (ASCII)
# =============================================================================
attacks = ["clean", "fgsm", "pgd20", "pgd100", "cw"]
atk_labels = {
    "clean":  "Clean",
    "fgsm":   "FGSM",
    "pgd20":  "PGD-20",
    "pgd100": "PGD-100",
    "cw":     "C&W-L2",
}

def format_table(eps_label, eps_val):
    lines = []
    lines.append(f"\nTable 1: Accuracy (%) under adversarial attacks "
                 f"[ε={eps_val:.2f}, normalised MFCC space]")
    lines.append(f"{'Model':<28}" +
                 "".join(f"{atk_labels[a]:>10}" for a in attacks))
    lines.append("-" * (28 + 10 * len(attacks)))
    for mname in models:
        res = all_results[eps_label][mname]
        row = f"{mname:<28}"
        for a in attacks:
            row += f"{res[a]:>10.2f}"
        lines.append(row)
    lines.append("-" * (28 + 10 * len(attacks)))
    # Robustness gain: M3 vs M1 on PGD-20
    m1_pgd20 = all_results[eps_label]["M1-Undefended"]["pgd20"]
    m3_pgd20 = all_results[eps_label]["M3-LSinc (Ours)"]["pgd20"]
    lines.append(f"\nRobustness gain (M3 vs M1, PGD-20): "
                 f"{m3_pgd20 - m1_pgd20:+.2f}%")
    return "\n".join(lines)

table_train = format_table("eps_train", EPS_TRAIN)
table_calib = format_table("eps_calib", EPS_CALIB)
full_table  = table_train + "\n\n" + table_calib

print("\n" + full_table)

with open(OUT_TXT, "w") as f:
    f.write(full_table)

# =============================================================================
# 11. SAVE JSON
# =============================================================================
with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)

# =============================================================================
# 12. PLOT TABLE AS FIGURE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {"M1-Undefended": "#d62728",
          "M2-ClassicalLifter": "#ff7f0e",
          "M3-LSinc (Ours)": "#2ca02c"}
x      = np.arange(len(attacks))
width  = 0.25

for ax_idx, (eps_label, eps_val) in enumerate([("eps_train", EPS_TRAIN),
                                                ("eps_calib", EPS_CALIB)]):
    ax = axes[ax_idx]
    for i, (mname, mdl) in enumerate(models.items()):
        vals = [all_results[eps_label][mname][a] for a in attacks]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=mname, color=colors[mname], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([atk_labels[a] for a in attacks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Table 1: ε={eps_val:.2f} (normalised MFCC)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

plt.suptitle("Robustness Evaluation: Undefended vs Classical Lifter vs L-Sinc",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {OUT_PNG}")

# =============================================================================
# 13. SAVE TO GOOGLE DRIVE
# =============================================================================
print("\nSaving outputs to Google Drive...")
save_to_drive_many([OUT_JSON, OUT_TXT, OUT_PNG])
print("\nEvaluation complete.")
os.sync

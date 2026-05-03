# Learnable Cepstral Liftering for Adversarial Robustness in Audio Classification

Reproducibility code for the IEEE Signal Processing Letters submission:
**"Learnable Cepstral Liftering for Adversarial Robustness in Audio Classification"**

---

## Repository Structure

| Program | Role | Produces |
|---|---|---|
| `1_diagnostic_experiment.py` | Quefrency energy profile diagnostic on UrbanSound8K | `fig1_quefrency_profile_fixed.png`, `01_quefrency_results.json` |
| `2_diagnostic_speech_commands.py` | SNR ratio analysis on Speech Commands v2; produces raw quefrency JSON used by 02b | `fig1_quefrency_profile_scv2.png`, `02_quefrency_results_scv2.json` |
| `2b_plot_snr_profile.py` | Generates Fig. 1 of the paper: two-panel SNR ratio + curve collapse plot from 02 JSON output | `fig1_snr_profile.png` |
| `3_fgsm_large_sample_verification.py` | Formant transition divergence verification (Proposition 2); large-sample FGSM vs PGD comparison | `large_sample_verification.png`, verification stats JSON |
| `4_full_pipeline.py` | Full training pipeline: Stage 1 clean pre-training + Stage 2 L-Sinc adversarial training with session-resume | `04_stage1_checkpoint.pth`, `04_defended_model.pth`, `04_fig3_lifter_profiles.png`, `04_stage1_curve.png`, `04_stage2_curve.png`, `04_stage1_stats.json`, `04_stage2_stats.json` |
| `5_evaluation_table1.py` | Evaluates M1 (undefended) and M3 (L-Sinc AT) across Clean, FGSM, PGD-20, PGD-100 | `05_table1.json`, `05_table1.txt`, `05_table1.png` |
| `5b_evaluation_table1.py` | Evaluates M1 (undefended) and M2 (PGD-AT, no lifter) across Clean, FGSM, PGD-20, PGD-100 | `05b_table1.json`, `05b_table1.txt`, `05b_table1.png` |
| `6_pgdat_baseline.py` | Trains PGD-AT baseline: identical architecture to defended model, adversarially trained without L-Sinc lifter | `06_pgdat_model.pth`, `06_pgdat_stage2_curve.png`, `06_pgdat_stage2_stats.json` |

---

## Dependencies

```
torch>=2.0
torchaudio>=2.0
numpy
matplotlib
tqdm
```

All scripts were developed and tested on Google Colab with a T4 GPU.
Python 3.12. No local GPU required for diagnostic scripts (01, 02, 03).

---

## Datasets

| Dataset | Used in | Download |
|---|---|---|
| Speech Commands v2 | 02, 10, 11, 11b, 12 | `torchaudio.datasets.SPEECHCOMMANDS` or manual download |
| UrbanSound8K | 01 | https://urbansounddataset.weebly.com/urbansound8k.html |

Scripts expect Speech Commands v2 as `SpeechCommandsV2.zip` placed at
`/content/drive/MyDrive/datasets/SpeechCommandsV2.zip` (Google Drive).
On a fresh Colab session the scripts mount Drive and copy the zip automatically.
UrbanSound8K expects `UrbanSound8K.tar.gz` at the same Drive location.

---

## Execution Order

Run scripts in the following order to reproduce all paper results:

### Step 1 — Diagnostic experiments (Figures 1 and 2, Propositions 1 and 2)
```bash
python 1_diagnostic_experiment.py          # UrbanSound8K quefrency profile
python 2_diagnostic_speech_commands.py     # Speech Commands v2 SNR ratio
python 2b_plot_snr_profile.py            # produces fig1_snr_profile.png (Fig. 1)
python 3_fgsm_large_sample_verification.py # Proposition 2 verification
```

### Step 2 — Train defended model (Section IV-B, Figure 3)
```bash
python 4_full_pipeline.py
```
Runs Stage 1 clean pre-training (15 epochs) followed by Stage 2 L-Sinc
adversarial training (27 epochs). Saves checkpoints every 3 epochs.
On session disconnect, re-running resumes automatically from the latest
checkpoint on the FTP server. Expected runtime: ~90–120 minutes on T4.

### Step 3 — Train PGD-AT baseline (Table 1, M2)
```bash
python 6_pgdat_baseline.py
```
Trains the standard PGD-AT baseline (same architecture, no lifter).
Loads Stage 1 weights from the checkpoint produced in Step 2.
Expected runtime: ~60–75 minutes on T4.

### Step 4 — Evaluate (Table 1)
```bash
python 5_evaluation_table1.py    # evaluates M1 (undefended) and M3 (L-Sinc AT)
python 5b_evaluation_table1.py   # evaluates M1 (undefended) and M2 (PGD-AT)
```
Both scripts download required checkpoints from FTP if not present locally.
Combined results from both scripts constitute Table 1 of the paper.

---


## Reproducing Table 1

After running Steps 2–4, Table 1 results are in:

| File | Contents |
|---|---|
| `05_table1.txt` | M1 (Undefended) and M3 (L-Sinc AT) results |
| `05b_table1.txt` | M1 (Undefended) and M2 (PGD-AT) results |

Combine both files to obtain the full three-row Table 1 as reported
in the paper.

---

## Reproducing Figures

| Figure | Source file | Generating Script |
|---|---|---|
| Fig. 1 (SNR ratio profile) | `fig1_snr_profile.png` | `2b_plot_snr_profile.py` |
| Fig. 2 (Lifter profiles) | `04_fig3_lifter_profiles.png` | `4_full_pipeline.py` |

---


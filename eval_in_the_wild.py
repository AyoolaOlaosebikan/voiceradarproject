"""
Out-of-domain evaluation on the In-the-Wild dataset (Muller et al. 2022).
Tests the model trained on ASVspoof 2019 LA on real-world audio it has never seen.

Run: python -u eval_in_the_wild.py
"""

import csv
import os
import random
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path
from transformers import HubertModel, AutoFeatureExtractor

from micro_frequencies import compute_micro_frequencies
from train import VoiceRadarMLP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIR  = Path("/Users/ayoola/Downloads/release_in_the_wild")
META_FILE    = DATASET_DIR / "meta.csv"
MAX_SAMPLES  = 500   # subsample to keep runtime reasonable (~10 min)
RANDOM_SEED  = 42

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

print("Loading HuBERT...")
extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert    = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
hubert.eval()

print("Loading VoiceRadar MLP...")
mlp = VoiceRadarMLP(input_dim=768).to(DEVICE)
mlp.load_state_dict(torch.load("voiceradar_best.pt", map_location=DEVICE))
mlp.eval()

# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------

all_samples = []
with open(META_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = 1 if row["label"] == "bona-fide" else 0
        all_samples.append((row["file"], label))

random.seed(RANDOM_SEED)
bonafide = [s for s in all_samples if s[1] == 1]
spoof    = [s for s in all_samples if s[1] == 0]
half     = MAX_SAMPLES // 2
samples  = random.sample(bonafide, min(half, len(bonafide))) + \
           random.sample(spoof,    min(half, len(spoof)))
random.shuffle(samples)

print(f"Evaluating {len(samples)} samples "
      f"({sum(1 for _,l in samples if l==1)} bonafide, "
      f"{sum(1 for _,l in samples if l==0)} spoof)...")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_audio_ffmpeg(path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        import soundfile as sf
        samples, _ = sf.read(out_path, dtype="float32")
        return samples
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


tp = fp = tn = fn = errors = 0

for i, (fname, label) in enumerate(samples):
    path = DATASET_DIR / fname
    try:
        waveform = load_audio_ffmpeg(path)
        inputs   = extractor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            E = hubert(inputs.input_values.to(DEVICE)).last_hidden_state.cpu().numpy()

        mf      = compute_micro_frequencies(E)
        emb_vec = torch.from_numpy(E[0].mean(axis=0).astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = mlp(emb_vec).item()

        predicted = 1 if prob >= 0.5 else 0

        if predicted == 1 and label == 1: tp += 1
        elif predicted == 1 and label == 0: fp += 1
        elif predicted == 0 and label == 0: tn += 1
        elif predicted == 0 and label == 1: fn += 1

    except Exception as e:
        errors += 1

    if (i + 1) % 50 == 0:
        done = tp + fp + tn + fn
        acc  = (tp + tn) / done if done > 0 else 0
        print(f"  {i+1}/{len(samples)}  acc so far: {acc:.4f}  errors: {errors}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

total   = tp + fp + tn + fn
acc     = (tp + tn) / total if total > 0 else 0
tpr     = tp / (tp + fn)    if (tp + fn) > 0 else 0
tnr     = tn / (tn + fp)    if (tn + fp) > 0 else 0
fpr     = 1 - tnr
fnr     = 1 - tpr
# EER approximation: point where FPR ≈ FNR
eer_approx = (fpr + fnr) / 2

print(f"\n{'='*45}")
print(f"In-the-Wild out-of-domain evaluation")
print(f"{'='*45}")
print(f"  Total evaluated:  {total}  (errors skipped: {errors})")
print(f"  Accuracy:         {acc:.4f}")
print(f"  TPR (bonafide):   {tpr:.4f}")
print(f"  TNR (spoof):      {tnr:.4f}")
print(f"  EER (approx):     {eer_approx:.4f}  ({eer_approx*100:.1f}%)")
print(f"\n  Compare: paper reports Whisper Features EER=26.72% on this dataset")
print(f"           wav2vec 2.0 EER=0.82% on ASVspoof 2021 (different domain)")

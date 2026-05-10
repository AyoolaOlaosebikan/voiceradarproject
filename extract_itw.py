"""
Extract HuBERT embeddings for the In-the-Wild dataset.
Subsamples to MAX_SAMPLES balanced bonafide/spoof to keep runtime manageable.
Saves to embeddings/itw/<filename>.npy  (shape [T, 768])
Safe to interrupt and re-run — skips files already done.
"""

import csv
import os
import random
import subprocess
import tempfile
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from transformers import HubertModel, AutoFeatureExtractor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIR = Path("/Users/ayoola/Downloads/release_in_the_wild")
META_FILE   = DATASET_DIR / "meta.csv"
EMB_DIR     = Path("embeddings/itw")
MAX_SAMPLES = 6000   # 3000 bonafide + 3000 spoof
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

DEVICE = "cpu"  # MPS causes OOM on 8GB M2 over long runs; CPU is slower but stable
print(f"Device: {DEVICE}")

extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model     = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
model.eval()

# ---------------------------------------------------------------------------
# Subsample
# ---------------------------------------------------------------------------

all_samples = []
with open(META_FILE) as f:
    for row in csv.DictReader(f):
        label = 1 if row["label"] == "bona-fide" else 0
        all_samples.append((row["file"], label))

random.seed(RANDOM_SEED)
bonafide = [s for s in all_samples if s[1] == 1]
spoof    = [s for s in all_samples if s[1] == 0]
half     = MAX_SAMPLES // 2
samples  = random.sample(bonafide, min(half, len(bonafide))) + \
           random.sample(spoof,    min(half, len(spoof)))

EMB_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nExtracting {len(samples)} files → {EMB_DIR}")

# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

def get_embedding(wav_path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path),
             "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        audio, _ = sf.read(out_path, dtype="float32")
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            E = model(inputs.input_values.to(DEVICE)).last_hidden_state
        return E.squeeze(0).cpu().numpy()   # [T, 768]
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


processed = skipped = errors = 0

for i, (fname, label) in enumerate(samples):
    stem     = Path(fname).stem
    npy_path = EMB_DIR / f"{stem}.npy"

    if npy_path.exists():
        skipped += 1
        continue

    try:
        emb = get_embedding(DATASET_DIR / fname)
        np.save(npy_path, emb)
        processed += 1
    except Exception as e:
        print(f"  [error] {fname}: {e}")
        errors += 1

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(samples)}  processed={processed}  "
              f"skipped={skipped}  errors={errors}", flush=True)
        if DEVICE == "mps":
            torch.mps.empty_cache()

print(f"\nDone. processed={processed}  skipped={skipped}  errors={errors}")

# Save the label manifest so precompute_fo knows which stem→label
manifest = {Path(f).stem: (1 if l == 1 else 0) for f, l in samples}
import json
with open(EMB_DIR / "labels.json", "w") as f:
    json.dump(manifest, f)
print(f"Label manifest saved to {EMB_DIR}/labels.json")

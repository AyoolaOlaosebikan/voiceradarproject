"""
Stage 4a: Extract HuBERT embeddings for ASVspoof 2019 LA and save to disk.

Processes one file at a time to stay within 8GB memory. Skips files that
already have a saved embedding so the script is safe to re-run after interruption.

Output layout:
  embeddings/train/<filename>.npy   shape [T, 768]
  embeddings/dev/<filename>.npy
"""

import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from transformers import HubertModel, AutoFeatureExtractor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("/Users/ayoola/Downloads/asvspoof/LA/LA")
AUDIO_DIRS = {
    "train": DATASET_ROOT / "ASVspoof2019_LA_train" / "flac",
    "dev":   DATASET_ROOT / "ASVspoof2019_LA_dev"   / "flac",
    "eval":  DATASET_ROOT / "ASVspoof2019_LA_eval"  / "flac",
}
PROTOCOL_FILES = {
    "train": DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
}
EMB_ROOT = Path("embeddings")

# Subsample to keep memory and time manageable on M2.
# Set to None to process everything.
MAX_PER_SPLIT = {"train": 10000, "dev": 2000, "eval": 3000}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/hubert-base-ls960"

device = "cpu"  # MPS causes OOM on 8GB M2 over long runs; CPU is slower but stable
print(f"Device: {device}")

extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model     = HubertModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def get_embedding(flac_path: Path) -> np.ndarray:
    """Load one .flac file and return HuBERT embedding as [T, 768] float32."""
    waveform_np, sr = sf.read(flac_path)
    waveform = torch.from_numpy(waveform_np).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.input_values.to(device))

    return outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [T, 768]


def parse_protocol(protocol_file: Path):
    """Return list of (filename, label_int) from a CM protocol file.
    label: 1 = bonafide (human), 0 = spoof (AI)
    """
    entries = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.strip().split()
            fname, label_str = parts[1], parts[4]
            label = 1 if label_str == "bonafide" else 0
            entries.append((fname, label))
    return entries


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

for split in ("train", "dev", "eval"):
    entries = parse_protocol(PROTOCOL_FILES[split])
    max_n = MAX_PER_SPLIT.get(split)
    if max_n is not None:
        # Keep a balanced subsample: equal bonafide and spoof where possible
        bonafide = [e for e in entries if e[1] == 1]
        spoof    = [e for e in entries if e[1] == 0]
        half     = max_n // 2
        entries  = bonafide[:half] + spoof[:half]

    out_dir = EMB_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{split}: {len(entries)} files → {out_dir}")
    skipped = 0
    processed = 0
    errors = 0

    for i, (fname, label) in enumerate(entries):
        npy_path = out_dir / f"{fname}.npy"
        if npy_path.exists():
            skipped += 1
            continue

        flac_path = AUDIO_DIRS[split] / f"{fname}.flac"
        if not flac_path.exists():
            print(f"  [missing] {flac_path}")
            errors += 1
            continue

        try:
            emb = get_embedding(flac_path)
            np.save(npy_path, emb)
            processed += 1
        except Exception as e:
            print(f"  [error] {fname}: {e}")
            errors += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(entries)}  processed={processed}  skipped={skipped}  errors={errors}")
            if device == "mps":
                torch.mps.empty_cache()

    print(f"  Done. processed={processed}  skipped={skipped}  errors={errors}")

print("\nEmbedding extraction complete.")

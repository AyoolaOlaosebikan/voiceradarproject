"""
Precompute micro-frequencies and f_o for all saved embeddings.

Saves a single file per split:
  embeddings/train_fo.npz  — keys: filenames, emb_mean, fo, label
  embeddings/dev_fo.npz
  embeddings/eval_fo.npz
  embeddings/itw_fo.npz   — In-the-Wild, labels read from embeddings/itw/labels.json

Run once after extract_embeddings.py (and extract_itw.py for itw).
train.py will use these cached values.
"""

import json
import numpy as np
from pathlib import Path
from micro_frequencies import compute_micro_frequencies
from doppler import compute_fo

EMB_ROOT     = Path("embeddings")
DATASET_ROOT = Path("/Users/ayoola/Downloads/asvspoof/LA/LA")
PROTOCOLS    = {
    "train": DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
}


def parse_protocol(path):
    labels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            labels[parts[1]] = 1 if parts[4] == "bonafide" else 0
    return labels


def process_split(split, emb_dir, labels):
    out_path = EMB_ROOT / f"{split}_fo.npz"
    if out_path.exists():
        print(f"{split}: already exists, skipping.")
        return

    paths = sorted(emb_dir.glob("*.npy"))
    n     = len(paths)
    print(f"\n{split}: precomputing f_o for {n} files...")

    filenames = []
    emb_means = []
    fo_vals   = []
    label_arr = []
    fs_vals   = []
    var_vals  = []

    for i, p in enumerate(paths):
        fname = p.stem
        label = labels.get(fname, -1)
        if label == -1:
            continue

        E   = np.load(p)                          # [T, 768]
        mf  = compute_micro_frequencies(E)
        fo  = compute_fo(mf["delta_f_total"], mf["variance"], float(label))
        emb_mean = E.mean(axis=0).astype(np.float32)

        filenames.append(fname)
        emb_means.append(emb_mean)
        fo_vals.append(fo)
        label_arr.append(label)
        fs_vals.append(mf["delta_f_total"])
        var_vals.append(mf["variance"])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}", flush=True)

    np.savez(
        out_path,
        filenames = np.array(filenames),
        emb_mean  = np.array(emb_means, dtype=np.float32),
        fo        = np.array(fo_vals,   dtype=np.float64),
        label     = np.array(label_arr, dtype=np.int32),
        fs        = np.array(fs_vals,   dtype=np.float64),
        variance  = np.array(var_vals,  dtype=np.float64),
    )
    print(f"  Saved {out_path}  ({len(filenames)} samples)")


# --- ASVspoof splits ---
for split in ("train", "dev", "eval"):
    process_split(split, EMB_ROOT / split, parse_protocol(PROTOCOLS[split]))

# --- In-the-Wild ---
itw_emb_dir    = EMB_ROOT / "itw"
itw_labels_file = itw_emb_dir / "labels.json"

if itw_labels_file.exists():
    with open(itw_labels_file) as f:
        itw_labels = json.load(f)   # {stem: 0|1}
    process_split("itw", itw_emb_dir, itw_labels)
else:
    print("\nitw: labels.json not found — run extract_itw.py first, skipping.")

print("\nDone.")

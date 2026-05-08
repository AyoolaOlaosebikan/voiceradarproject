"""
Stage 4b: Train the VoiceRadar MLP on pre-extracted embeddings.

Architecture (Section IV-B):
  6 fully-connected layers: 512 → 256 → 128 → 64 → 32 → 1
  ReLU activations, sigmoid output
  Adam optimizer, batch size 64

Loss (Equation 2):
  loss = BCE + 0.6 * (fo(x, y) - fo(x, F(E(x))))

Run extract_embeddings.py (and extract_itw.py) then precompute_fo.py first.
Set USE_COMBINED_TRAINING = True to include In-the-Wild data in training.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMB_ROOT             = Path("embeddings")
BATCH_SIZE           = 64
EPOCHS               = 30
LR                   = 1e-4
DEVICE               = "mps" if torch.backends.mps.is_available() else "cpu"
USE_COMBINED_TRAINING = True   # set False to train on ASVspoof only

print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Dataset — loads from precomputed .npz cache (run precompute_fo.py first)
# ---------------------------------------------------------------------------

class VoiceDataset(Dataset):
    def __init__(self, split: str):
        npz_path = EMB_ROOT / f"{split}_fo.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} not found — run precompute_fo.py first")
        data = np.load(npz_path)
        self.emb_mean = data["emb_mean"]                      # [N, 768]
        self.fo       = data["fo"].astype(np.float32)         # [N]
        self.fs       = data["fs"].astype(np.float32)         # [N]
        self.variance = data["variance"].astype(np.float32)   # [N]
        self.label    = data["label"]                         # [N]
        print(f"  {split}: {len(self.label)} samples loaded")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.emb_mean[idx]),
            torch.tensor(self.label[idx],    dtype=torch.float32),
            torch.tensor(self.fo[idx],       dtype=torch.float32),
            torch.tensor(self.fs[idx],       dtype=torch.float32),
            torch.tensor(self.variance[idx], dtype=torch.float32),
        )


# Backwards-compatible alias used by app.py / eval scripts
ASVspoofDataset = VoiceDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VoiceRadarMLP(nn.Module):
    """6-layer MLP matching paper Section IV-B."""
    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [B]


# ---------------------------------------------------------------------------
# Loss (Equation 2)
# ---------------------------------------------------------------------------

FJ0_0_1 = 2.404826  # first zero of J0, constant


def voiceradar_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    fo_true: torch.Tensor,
    fs: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """
    loss = BCE + 0.6 * mean(fo(true) - fo(predicted))   [Equation 2]

    fo(predicted) recomputes the Doppler equation using pred as y, so gradient
    flows through it. Normalized by fo_true so the physics term stays O(1)
    regardless of the absolute magnitude of fo values (~1M).
    Logged in notes/divergences.md as D7.
    """
    bce = nn.functional.binary_cross_entropy(pred, label)

    # fo(predicted): same Doppler formula but y = pred (continuous)
    vs_pred  = pred * variance.abs()
    cv       = variance * FJ0_0_1
    fo_pred  = (cv / (cv - vs_pred)) * fs

    # Normalize so physics term is dimensionless
    physics_term = ((fo_true - fo_pred) / (fo_true.abs() + 1e-8)).mean()

    return bce + 0.6 * physics_term


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

print("Loading datasets...")
asv_train_ds = VoiceDataset("train")
dev_ds       = VoiceDataset("dev")
eval_ds      = VoiceDataset("eval")

itw_path = EMB_ROOT / "itw_fo.npz"
if USE_COMBINED_TRAINING and itw_path.exists():
    itw_ds   = VoiceDataset("itw")
    train_ds = ConcatDataset([asv_train_ds, itw_ds])
    print(f"  Combined train: {len(train_ds)} samples "
          f"(ASVspoof {len(asv_train_ds)} + ITW {len(itw_ds)})")
else:
    train_ds = asv_train_ds
    if USE_COMBINED_TRAINING:
        print("  itw_fo.npz not found — training on ASVspoof only")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

model     = VoiceRadarMLP(input_dim=768).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_dev_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    train_loss = 0.0
    for emb, label, fo_true, fs, variance in train_loader:
        emb, label = emb.to(DEVICE), label.to(DEVICE)
        fo_true    = fo_true.to(DEVICE)
        fs         = fs.to(DEVICE)
        variance   = variance.to(DEVICE)

        pred = model(emb)
        loss = voiceradar_loss(pred, label, fo_true, fs, variance)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --- eval ---
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for emb, label, fo_true, fs, variance in dev_loader:
            emb, label = emb.to(DEVICE), label.to(DEVICE)
            pred = model(emb)
            predicted = (pred >= 0.5).float()
            correct += (predicted == label).sum().item()
            total   += label.size(0)

    dev_acc = correct / total
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), "voiceradar_best.pt")

    print(f"Epoch {epoch:2d}/{EPOCHS}  loss={train_loss:.4f}  dev_acc={dev_acc:.4f}  best={best_dev_acc:.4f}")

print(f"\nTraining complete. Best dev accuracy: {best_dev_acc:.4f}")

# --- final eval on held-out eval split (unseen attack types) ---
print("\nEvaluating on eval split (unseen attack types)...")
model.load_state_dict(torch.load("voiceradar_best.pt", map_location=DEVICE))
model.eval()

correct = total = tp = fp = tn = fn = 0
with torch.no_grad():
    for emb, label, fo_true, fs, variance in eval_loader:
        emb, label = emb.to(DEVICE), label.to(DEVICE)
        pred = model(emb)
        predicted = (pred >= 0.5).float()
        correct += (predicted == label).sum().item()
        total   += label.size(0)
        tp += ((predicted == 1) & (label == 1)).sum().item()
        fp += ((predicted == 1) & (label == 0)).sum().item()
        tn += ((predicted == 0) & (label == 0)).sum().item()
        fn += ((predicted == 0) & (label == 1)).sum().item()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"  Eval accuracy: {correct/total:.4f}  ({correct}/{total})")
print(f"  TPR (bonafide detected): {tpr:.4f}")
print(f"  TNR (spoof detected):    {tnr:.4f}")

# --- out-of-domain eval on In-the-Wild (if available) ---
if itw_path.exists():
    print("\nEvaluating on In-the-Wild (out-of-domain)...")
    itw_loader = DataLoader(VoiceDataset("itw"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    correct = total = tp = fp = tn = fn = 0
    with torch.no_grad():
        for emb, label, fo_true, fs, variance in itw_loader:
            emb, label = emb.to(DEVICE), label.to(DEVICE)
            pred = model(emb)
            predicted = (pred >= 0.5).float()
            correct += (predicted == label).sum().item()
            total   += label.size(0)
            tp += ((predicted == 1) & (label == 1)).sum().item()
            fp += ((predicted == 1) & (label == 0)).sum().item()
            tn += ((predicted == 0) & (label == 0)).sum().item()
            fn += ((predicted == 0) & (label == 1)).sum().item()
    tpr_itw = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr_itw = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_itw = 1 - tnr_itw
    fnr_itw = 1 - tpr_itw
    eer_itw = (fpr_itw + fnr_itw) / 2
    print(f"  ITW accuracy: {correct/total:.4f}  ({correct}/{total})")
    print(f"  TPR (bonafide): {tpr_itw:.4f}")
    print(f"  TNR (spoof):    {tnr_itw:.4f}")
    print(f"  EER (approx):   {eer_itw*100:.1f}%  (baseline: 38.6% ASVspoof-only)")
